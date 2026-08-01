using AbstractCosmologicalEmulators
using DataFrames
using EmulatorsTrainer
using JSON3
using NPZ
using SimpleChains

const INPUT_COLUMNS = [:ln10As, :ns, :tau, :H0, :omega_b, :omega_c]
const SPLIT_SEED = 20260733
const INITIALIZATION_SEED = 20260734

amplitude_factor(parameters, spectrum) = spectrum == "PP" ?
    exp(parameters["ln10As"]) * 1.0e-10 :
    exp(parameters["ln10As"]) * 1.0e-10 * exp(-2 * parameters["tau"])

function write_postprocessing(directory, spectrum)
    inverse_julia = spectrum == "PP" ? "exp.(output)" : "output"
    inverse_python = spectrum == "PP" ? "jnp.exp(output)" : "output"
    tau_julia = spectrum == "PP" ? "one(eltype(input))" : "exp(-2 * input[3])"
    tau_julia_batch = spectrum == "PP" ? "one(eltype(input))" : "exp.(-2 .* input[3:3, :])"
    tau_python = spectrum == "PP" ? "1.0" : "jnp.exp(-2 * input[..., 2])"
    write(joinpath(directory, "postprocessing.jl"), """(input, output, emu) -> begin
    factor = if ndims(input) == 1
        exp(input[1]) * 1.0e-10 * $tau_julia
    else
        exp.(input[1:1, :]) .* 1.0e-10 .* $tau_julia_batch
    end
    $inverse_julia .* factor
end
""")
    write(joinpath(directory, "postprocessing.py"), """import jax.numpy as jnp

def postprocessing(input, output):
    factor = jnp.exp(input[..., 0]) * 1.0e-10 * $tau_python
    restored = $inverse_python
    return restored * factor[..., None] if output.ndim == 2 else restored * factor
""")
end

function main()
    length(ARGS) >= 1 || error("Usage: train.jl SPECTRUM [DATA_DIRECTORY] [OUTPUT_DIRECTORY]")
    spectrum = uppercase(ARGS[1])
    spectrum in ("TT", "TE", "EE", "PP") || error("Unknown spectrum: $spectrum")
    data_directory = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "data", "camb_lcdm_500", "dataset.h5")
    output_root = length(ARGS) >= 3 ? abspath(ARGS[3]) : joinpath(@__DIR__, "artifacts", "camb_lcdm_500")
    output_directory = joinpath(output_root, spectrum)
    node_count = spectrum == "PP" ? 192 : 256

    dataset = EmulatorsTrainer.load_hdf5_dataset(data_directory)
    all(dataset.valid) || error("HDF5 dataset contains invalid samples")
    parameters_array = dataset.parameters
    parameter_names = dataset.parameter_names
    observable = get(dataset.observables, Symbol(spectrum), nothing)
    observable === nothing && error("Observable $spectrum is not present in $data_directory")
    node_count == size(observable, 2) || error("Expected $node_count nodes, got $(size(observable, 2))")

    frame = DataFrame(
        sample_id=String[],
        ln10As=Float64[], ns=Float64[], tau=Float64[], H0=Float64[],
        omega_b=Float64[], omega_c=Float64[], observable=Vector{Float64}[],
    )
    for sample_index in axes(parameters_array, 1)
        parameters = Dict(parameter_names[j] => parameters_array[sample_index, j]
            for j in axes(parameters_array, 2))
        sample_id = "sample_$(lpad(sample_index, 6, '0'))"
        target = spectrum == "PP" ? log.(observable[sample_index, :] ./ amplitude_factor(parameters, spectrum)) :
            observable[sample_index, :] ./ amplitude_factor(parameters, spectrum)
        push!(frame, (
            sample_id=sample_id,
            ln10As=Float64(parameters["ln10As"]),
            ns=Float64(parameters["ns"]),
            tau=Float64(parameters["tau"]),
            H0=Float64(parameters["H0"]),
            omega_b=Float64(parameters["omega_b"]),
            omega_c=Float64(parameters["omega_c"]),
            observable=target,
        ))
    end
    load_report = (loaded=size(parameters_array, 1), skipped=0)
    println("Loaded $(load_report.loaded), skipped $(load_report.skipped) samples")

    input_limits = get_minmax_in(frame, INPUT_COLUMNS)
    _, output_array = extract_input_output_df(frame; input_columns=INPUT_COLUMNS)
    output_limits = get_minmax_out(output_array)
    maximin_df!(frame, input_limits, output_limits; input_columns=INPUT_COLUMNS)
    x_train, y_train, x_validation, y_validation, train_indices, validation_indices =
        getdata(
            frame;
            test_fraction=0.2,
            seed=SPLIT_SEED,
            input_columns=INPUT_COLUMNS,
            return_indices=true,
        )

    network_dictionary = Dict{String,Any}(
        "n_input_features" => length(INPUT_COLUMNS),
        "n_output_features" => node_count,
        "n_hidden_layers" => 5,
        "layers" => Dict(
            "layer_$index" => Dict("n_neurons" => 64, "activation_function" => "tanh")
            for index in 1:5
        ),
        "emulator_description" => Dict(
            "author" => "Marco Bonici",
            "author_email" => "bonici.marco@gmail.com",
            "spectrum" => spectrum,
            "parameters" => join(string.(INPUT_COLUMNS), ", "),
            "source" => "CAMB ACT-DR6 precision LCDM",
            "representation" => "Chebyshev-Lobatto nodes",
            "output_transform" => spectrum == "PP" ? "log" : "linear",
        ),
    )
    network = AbstractCosmologicalEmulators._get_nn_simplechains(network_dictionary)
    steps_per_session = parse(Int, get(ENV, "CAPSE_STEPS_PER_SESSION", "2000"))
    sessions_per_rate = parse(Int, get(ENV, "CAPSE_SESSIONS_PER_RATE", "10"))
    batch_size = parse(Int, get(ENV, "CAPSE_BATCH_SIZE", "256"))
    config = SimpleChainsTrainingConfig(
        learning_rates=[1.0e-4, 7.0e-5, 5.0e-5, 2.0e-5, 1.0e-5,
            7.0e-6, 5.0e-6, 2.0e-6, 1.0e-6, 7.0e-7],
        sessions_per_rate=sessions_per_rate,
        steps_per_session=steps_per_session,
        batch_size=batch_size,
        initialization_seed=INITIALIZATION_SEED,
    )
    callback = progress -> println(
        "steps=$(progress.total_steps) lr=$(progress.learning_rate) " *
        "train=$(progress.training_loss) validation=$(progress.validation_loss) " *
        "best=$(progress.best_validation_loss)",
    )
    result = train_simplechains(
        network, x_train, y_train, x_validation, y_validation;
        config, callback,
        checkpoint_callback=(parameters, progress) ->
            save_training_checkpoint(output_directory, parameters, progress),
    )

    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
    npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
    grid_name = spectrum == "PP" ? :ell_192 : :ell_256
    haskey(dataset.axes, grid_name) || error("Missing axis $grid_name in HDF5 dataset")
    npzwrite(joinpath(output_directory, "l.npy"), dataset.axes[grid_name])
    npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
    npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
    open(joinpath(output_directory, "nn_setup.json"), "w") do stream
        JSON3.write(stream, network_dictionary)
    end
    write_postprocessing(output_directory, spectrum)
    metadata = Dict{String,Any}(
        "spectrum" => spectrum,
        "dataset_directory" => data_directory,
        "n_loaded" => load_report.loaded,
        "n_skipped" => load_report.skipped,
        "n_train" => length(train_indices),
        "n_validation" => length(validation_indices),
        "split_seed" => SPLIT_SEED,
        "input_columns" => string.(INPUT_COLUMNS),
        "train_sample_ids" => frame.sample_id[train_indices],
        "validation_sample_ids" => frame.sample_id[validation_indices],
        "node_count" => node_count,
        "output_transform" => spectrum == "PP" ? "log" : "linear",
    )
    save_training_result(output_directory, result; metadata)
    println("Best validation loss: $(result.best_validation_loss)")
    println("Artifact: $output_directory")
end

main()
