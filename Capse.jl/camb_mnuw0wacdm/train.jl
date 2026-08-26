ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "JULIA_PYTHONCALL_EXE", something(Sys.which("python"), "python"))

using AbstractCosmologicalEmulators
using DataFrames
using EmulatorsTrainer
using HDF5
using JSON3
using NPZ
using PyCall
using SimpleChains

const INPUT_COLUMNS = [
    :ln10As, :ns, :tau, :H0, :omega_b, :omega_c, :Mnu, :w0, :wa,
]
const SPLIT_SEED = 20260736
const INITIALIZATION_SEED = 20260737

amplitude_factor(parameters, spectrum) = spectrum == "PP" ?
    exp(parameters["ln10As"]) * 1.0e-10 :
    exp(parameters["ln10As"]) * 1.0e-10 * exp(-2 * parameters["tau"])

function write_postprocessing(directory, spectrum, log_target)
    use_log = spectrum == "PP" || log_target
    inverse_julia = use_log ? "exp.(output)" : "output"
    inverse_python = use_log ? "jnp.exp(output)" : "output"
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

function interpolate_dense_chunk(dense, ell_dense, nodes, cubic_spline)
    spline = cubic_spline(ell_dense, dense; axis=1)
    return convert(Matrix{Float64}, spline(nodes))
end

function lobatto_nodes(node_count; lower=2.0, upper=9500.0)
    node_count >= 2 || error("Lobatto node count must be at least 2")
    theta = range(0.0, π; length=node_count)
    nodes = 0.5 .* (lower + upper) .- 0.5 .* (upper - lower) .* cos.(theta)
    nodes[1] = lower
    nodes[end] = upper
    return nodes
end

function training_nodes(file, spectrum, node_count)
    grid_name = node_count == 192 ? "ell_192" : node_count == 256 ? "ell_256" : nothing
    if grid_name !== nothing && haskey(file, "axes/$grid_name")
        return Float64.(file["axes/$grid_name"][:])
    end
    return lobatto_nodes(node_count)
end

function load_camb_training_frame(path, spectrum, node_count, log_target)
    cubic_spline = pyimport("scipy.interpolate").CubicSpline
    h5open(path, "r") do file
        parameters_array = Array(file["parameters"][:, :])
        parameter_names = String.(file["parameter_names"][:])
        valid = Bool.(file["valid"][:])
        all(valid) || error("HDF5 dataset contains invalid samples")
        dense_dataset = file["observables/$(spectrum)_dense"]
        ell_dense = Float64.(file["axes/ell_dense"][:])
        nodes = training_nodes(file, spectrum, node_count)
        size(dense_dataset, 2) == length(ell_dense) || error("Dense axis length mismatch")
        length(nodes) == node_count || error("Training node count mismatch")

        n_samples = size(parameters_array, 1)
        targets = Matrix{Float64}(undef, n_samples, node_count)
        chunk_size = 256
        for first_index in 1:chunk_size:n_samples
            last_index = min(first_index + chunk_size - 1, n_samples)
            dense = Array(dense_dataset[first_index:last_index, :])
            targets[first_index:last_index, :] = interpolate_dense_chunk(
                dense, ell_dense, nodes, cubic_spline,
            )
        end

        result = DataFrame(
            sample_id=String[],
            ln10As=Float64[], ns=Float64[], tau=Float64[], H0=Float64[],
            omega_b=Float64[], omega_c=Float64[], Mnu=Float64[], w0=Float64[],
            wa=Float64[], observable=Vector{Float64}[],
        )
        for sample_index in axes(parameters_array, 1)
            parameters = Dict(parameter_names[j] => parameters_array[sample_index, j]
                for j in axes(parameters_array, 2))
            amplitude = amplitude_factor(parameters, spectrum)
            target = log_target || spectrum == "PP" ?
                log.(targets[sample_index, :] ./ amplitude) :
                targets[sample_index, :] ./ amplitude
            push!(result, (
                sample_id="sample_$(lpad(sample_index, 6, '0'))",
                ln10As=Float64(parameters["ln10As"]), ns=Float64(parameters["ns"]),
                tau=Float64(parameters["tau"]), H0=Float64(parameters["H0"]),
                omega_b=Float64(parameters["omega_b"]), omega_c=Float64(parameters["omega_c"]),
                Mnu=Float64(parameters["Mnu"]), w0=Float64(parameters["w0"]),
                wa=Float64(parameters["wa"]), observable=target,
            ))
        end
        return result
    end
end

function main()
    length(ARGS) >= 1 || error("Usage: train.jl SPECTRUM [DATA_DIRECTORY] [OUTPUT_DIRECTORY]")
    requested_spectrum = uppercase(ARGS[1])
    log_target = endswith(requested_spectrum, "_LOG")
    spectrum = log_target ? chop(requested_spectrum; tail=4) : requested_spectrum
    spectrum in ("TT", "TE", "EE", "BB", "PP") || error("Unknown spectrum: $requested_spectrum")
    data_directory = length(ARGS) >= 2 ? abspath(ARGS[2]) :
        joinpath(@__DIR__, "data", "camb_mnuw0wacdm_1000", "dataset.h5")
    output_root = length(ARGS) >= 3 ? abspath(ARGS[3]) :
        joinpath(@__DIR__, "artifacts", "camb_mnuw0wacdm_1000")
    # Keep artifact subdirectories canonical even when the requested spectrum
    # selects a log target (e.g. TT_LOG is stored in the TT directory).
    output_directory = joinpath(output_root, spectrum)
    default_node_count = spectrum == "PP" ? 192 : 256
    node_count = parse(Int, get(ENV, "CAPSE_NODE_COUNT", string(default_node_count)))

    frame = load_camb_training_frame(data_directory, spectrum, node_count, log_target)
    load_report = (loaded=nrow(frame), skipped=0)
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
            "source" => "CAMB ACT-DR6 precision Mnu-w0-waCDM",
            "constraint" => "w0 + wa < -0.5",
            "representation" => "Chebyshev-Lobatto nodes",
            "output_transform" => (spectrum == "PP" || log_target) ? "log" : "linear",
        ),
    )
    network = AbstractCosmologicalEmulators._get_nn_simplechains(network_dictionary)
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
    npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
    training_axis = h5open(data_directory, "r") do file
        training_nodes(file, spectrum, node_count)
    end
    npzwrite(joinpath(output_directory, "l.npy"), training_axis)
    npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
    npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
    open(joinpath(output_directory, "nn_setup.json"), "w") do stream
        JSON3.write(stream, network_dictionary)
    end
    write_postprocessing(output_directory, spectrum, log_target)
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
    callback = progress -> begin
        println(
            "steps=$(progress.total_steps) lr=$(progress.learning_rate) " *
            "train=$(progress.training_loss) validation=$(progress.validation_loss) " *
            "best=$(progress.best_validation_loss)",
        )
        flush(stdout)
    end
    result = train_simplechains(
        network, x_train, y_train, x_validation, y_validation;
        config, callback,
        checkpoint_callback=(parameters, progress) ->
            save_training_checkpoint(output_directory, parameters, progress),
    )

    metadata = Dict{String,Any}(
        "spectrum" => spectrum,
        "dataset_directory" => data_directory,
        "n_loaded" => load_report.loaded,
        "n_skipped" => load_report.skipped,
        "n_train" => length(train_indices),
        "n_validation" => length(validation_indices),
        "split_seed" => SPLIT_SEED,
        "input_columns" => string.(INPUT_COLUMNS),
        "constraint" => "w0 + wa < -0.5",
        "train_sample_ids" => frame.sample_id[train_indices],
        "validation_sample_ids" => frame.sample_id[validation_indices],
        "node_count" => node_count,
        "interpolation_method" => "SciPy CubicSpline on dense ell grid at training time",
        "interpolation_source" => "$(spectrum)_dense",
        "interpolation_grid" => node_count == 192 ? "ell_192" :
            node_count == 256 ? "ell_256" : "generated_lobatto_$node_count",
        "output_transform" => (spectrum == "PP" || log_target) ? "log" : "linear",
        "requested_spectrum" => requested_spectrum,
    )
    save_training_result(output_directory, result; metadata)
    println("Best validation loss: $(result.best_validation_loss)")
    println("Artifact: $output_directory")
end

main()
