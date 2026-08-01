using AbstractCosmologicalEmulators
using ArgParse
using DataFrames
using DelimitedFiles
using Effort
using EmulatorsTrainer
using JSON3
using NPZ

const INPUT_COLUMNS = [:z, :ln10As, :ns, :H0, :ombh2, :omch2, :Mν, :w0, :wa]

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--component"
        default = "loop"
        "--multipole", "-l"
        arg_type = Int
        default = 0
        "--path-input", "-i"
        required = true
        "--path-output", "-o"
        required = true
        "--split-seed"
        arg_type = Int
        default = 20260739
        "--initialization-seed"
        arg_type = Int
        default = 20260740
        "--steps-per-session"
        arg_type = Int
        default = 2_000
        "--sessions-per-rate"
        arg_type = Int
        default = 10
        "--batch-size"
        arg_type = Int
        default = 256
    end
    return parse_args(settings)
end

function component_columns(component)
    component == "11" && return 1:3
    component == "loop" && return 4:12
    component == "ct" && return 13:16
    throw(ArgumentError("component must be one of: 11, loop, ct"))
end

function growth_factor(parameters)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=3.0,
        nₛ=0.96,
        h=parameters["H0"] / 100,
        ωb=parameters["ombh2"],
        ωc=parameters["omch2"],
        mν=parameters["Mν"],
        w0=parameters["w0"],
        wa=parameters["wa"],
    )
    return Effort.D_z(parameters["z"], cosmology)
end

function preprocessing_factor(parameters, component)
    amplitude = exp(parameters["ln10As"]) * 1.0e-10
    factor = amplitude * growth_factor(parameters)^2
    return component == "loop" ? factor^2 : factor
end

function first_sample_directory(root)
    for (directory, _, files) in walkdir(root)
        "effort_dict.json" in files && return directory
    end
    error("No sample directory found under $root")
end

function copy_artifact_template(source_name, destination_name, output_directory)
    source = joinpath(@__DIR__, source_name)
    isfile(source) || error("Missing artifact template: $source")
    cp(source, joinpath(output_directory, destination_name); force=true)
end

function main()
    arguments = parse_commandline()
    component = arguments["component"]
    multipole = arguments["multipole"]
    multipole in (0, 2, 4) || throw(ArgumentError("multipole must be 0, 2, or 4"))
    columns = component_columns(component)
    input_directory = abspath(arguments["path-input"])
    output_directory = joinpath(abspath(arguments["path-output"]), string(multipole), component)
    mkpath(output_directory)

    dataset = EmulatorsTrainer.load_hdf5_dataset(input_directory)
    parameters_array = dataset.parameters
    parameter_names = dataset.parameter_names
    observable_name = Symbol("pk_$(multipole)")
    observable = get(dataset.observables, observable_name, nothing)
    observable === nothing && error("Observable $observable_name is not present in $input_directory")
    all(dataset.valid) || error("HDF5 dataset contains invalid samples")

    frame = DataFrame(
        sample_id=String[], z=Float64[], ln10As=Float64[], ns=Float64[],
        H0=Float64[], ombh2=Float64[], omch2=Float64[], Mν=Float64[],
        w0=Float64[], wa=Float64[], observable=Vector{Float64}[],
    )
    for sample_index in axes(parameters_array, 1)
        parameters = Dict(parameter_names[j] => parameters_array[sample_index, j]
            for j in axes(parameters_array, 2))
        selected = vec(Array(observable[sample_index, :, :])[:, columns]) ./ preprocessing_factor(parameters, component)
        push!(frame, (
            sample_id="sample_$(lpad(sample_index, 6, '0'))",
            z=Float64(parameters["z"]), ln10As=Float64(parameters["ln10As"]), ns=Float64(parameters["ns"]),
            H0=Float64(parameters["H0"]), ombh2=Float64(parameters["ombh2"]), omch2=Float64(parameters["omch2"]),
            Mν=Float64(parameters["Mν"]), w0=Float64(parameters["w0"]), wa=Float64(parameters["wa"]), observable=selected,
        ))
    end
    load_report = (loaded=size(frame, 1), skipped=0)
    load_report.loaded >= 2 || error("Too few valid samples to train")

    input_limits = get_minmax_in(frame, INPUT_COLUMNS)
    _, output_array = extract_input_output_df(frame; input_columns=INPUT_COLUMNS)
    output_limits = get_minmax_out(output_array)
    maximin_df!(frame, input_limits, output_limits; input_columns=INPUT_COLUMNS)
    x_train, y_train, x_validation, y_validation, train_indices, validation_indices =
        getdata(
            frame;
            test_fraction=0.2,
            seed=arguments["split-seed"],
            input_columns=INPUT_COLUMNS,
            return_indices=true,
        )

    n_output_features = size(y_train, 1)
    network_dictionary = Dict{String,Any}(
        "n_input_features" => length(INPUT_COLUMNS),
        "n_output_features" => n_output_features,
        "n_hidden_layers" => 5,
        "layers" => Dict(
            "layer_$index" => Dict("n_neurons" => 64, "activation_function" => "tanh")
            for index in 1:5
        ),
        "emulator_description" => Dict(
            "source" => "CLASS + Velocileptors REPT",
            "cosmology" => "Mnu-w0-waCDM",
            "component" => component,
            "multipole" => multipole,
            "parameters" => join(string.(INPUT_COLUMNS), ", "),
        ),
    )
    network = AbstractCosmologicalEmulators._get_nn_simplechains(network_dictionary)
    npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
    npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
    npzwrite(joinpath(output_directory, "k.npy"), vec(dataset.observables[:kv][1, :]))
    npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
    npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
    open(joinpath(output_directory, "nn_setup.json"), "w") do stream
        JSON3.write(stream, network_dictionary)
    end
    if component == "loop"
        copy_artifact_template("postprocessing_loop.py", "postprocessing.py", output_directory)
        copy_artifact_template("postprocessing_loop.jl", "postprocessing.jl", output_directory)
    else
        copy_artifact_template("postprocessing.py", "postprocessing.py", output_directory)
        copy_artifact_template("postprocessing.jl", "postprocessing.jl", output_directory)
    end
    copy_artifact_template("stochmodel_$(multipole).py", "stochmodel.py", output_directory)
    copy_artifact_template("stochmodel_$(multipole).jl", "stochmodel.jl", output_directory)
    config = SimpleChainsTrainingConfig(
        learning_rates=[1.0e-4, 7.0e-5, 5.0e-5, 2.0e-5, 1.0e-5,
            7.0e-6, 5.0e-6, 2.0e-6, 1.0e-6, 7.0e-7],
        sessions_per_rate=arguments["sessions-per-rate"],
        steps_per_session=arguments["steps-per-session"],
        batch_size=arguments["batch-size"],
        initialization_seed=arguments["initialization-seed"],
    )
    callback = progress -> begin
        println(
            "steps=$(progress.total_steps) train=$(progress.training_loss) " *
            "validation=$(progress.validation_loss) best=$(progress.best_validation_loss)",
        )
        flush(stdout)
    end
    checkpoint_callback = (parameters, progress) ->
        save_training_checkpoint(output_directory, parameters, progress)
    result = train_simplechains(
        network, x_train, y_train, x_validation, y_validation;
        config, callback, checkpoint_callback,
    )

    metadata = Dict{String,Any}(
        "component" => component,
        "multipole" => multipole,
        "n_loaded" => load_report.loaded,
        "n_skipped" => load_report.skipped,
        "n_train" => length(train_indices),
        "n_validation" => length(validation_indices),
        "input_columns" => string.(INPUT_COLUMNS),
        "train_sample_ids" => frame.sample_id[train_indices],
        "validation_sample_ids" => frame.sample_id[validation_indices],
    )
    save_training_result(output_directory, result; metadata)
    println("Best validation loss: $(result.best_validation_loss)")
end

main()
