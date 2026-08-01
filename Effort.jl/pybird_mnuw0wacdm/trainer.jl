using AbstractCosmologicalEmulators
using ArgParse
using DataFrames
using Effort
using EmulatorsTrainer
using JSON
using JSON3
using NPZ
using SimpleChains

const INPUT_COLUMNS = [:z, :ln10A_s, :ns, :H0, :omega_b, :omega_cdm, :Mν, :w0, :wa]

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--component"; default="11"
        "--multipole", "-l"; arg_type=Int; default=0
        "--path-input", "-i"; required=true
        "--path-output", "-o"; required=true
        "--preprocessing", "-p"; default="AsDzprec"
        "--split-seed"; arg_type=Int; default=20260764
        "--initialization-seed"; arg_type=Int; default=20260765
        "--steps-per-session"; arg_type=Int; default=1000
        "--sessions-per-rate"; arg_type=Int; default=10
        "--batch-size"; arg_type=Int; default=512
    end
    return parse_args(settings)
end

function component_columns(component)
    component == "11" && return (:P11l, 1:3)
    component == "loop" && return (:Ploopl, 1:12)
    component == "ct" && return (:Pctl, 1:6)
    throw(ArgumentError("component must be one of: 11, loop, ct"))
end

function preprocessing_factor(parameters, preprocessing, component)
    z = parameters["z"]
    h = parameters["H0"] / 100
    Ωcb0 = (parameters["ombh2"] + parameters["omch2"]) / h^2
    As = exp(parameters["ln10As"]) * 1.0e-10
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=3.0, nₛ=0.96, h=h, ωb=parameters["ombh2"], ωc=parameters["omch2"],
        mν=parameters["Mν"], w0=parameters["w0"], wa=parameters["wa"],
    )
    D = Effort.D_z(z, cosmology)
    base = preprocessing == "noprec" ? 1.0 :
        preprocessing == "Asprec" ? As :
        preprocessing == "Dzprec" ? D^2 :
        preprocessing == "AsDzprec" ? As * D^2 :
        throw(ArgumentError("Unknown preprocessing: $preprocessing"))
    return component == "loop" ? base^2 : base
end

function main()
    arguments = parse_commandline()
    component = arguments["component"]
    multipole = arguments["multipole"]
    multipole in (0, 2, 4) || throw(ArgumentError("multipole must be 0, 2, or 4"))
    observable_name, columns = component_columns(component)
    ell_index = multipole ÷ 2 + 1
    dataset = load_hdf5_dataset(abspath(arguments["path-input"]))
    all(dataset.valid) || error("HDF5 dataset contains invalid samples")
    haskey(dataset.observables, observable_name) || error("Missing $observable_name observable")
    observable = dataset.observables[observable_name]
    size(observable, 2) >= ell_index || error("Invalid multipole index")
    k = dataset.axes[:kd]
    n_samples = size(dataset.parameters, 1)

    frame = DataFrame(z=Float64[], ln10A_s=Float64[], ns=Float64[], H0=Float64[],
        omega_b=Float64[], omega_cdm=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[],
        observable=Vector{Float64}[])
    for sample_index in 1:n_samples
        parameters = Dict(dataset.parameter_names[j] => dataset.parameters[sample_index, j]
            for j in axes(dataset.parameters, 2))
        slice = Array(observable[sample_index, ell_index, columns, :])
        selected = vec(slice) ./ preprocessing_factor(
            parameters, arguments["preprocessing"], component,
        )
        push!(frame, (
            parameters["z"], parameters["ln10As"], parameters["ns"], parameters["H0"],
            parameters["ombh2"], parameters["omch2"], parameters["Mν"],
            parameters["w0"], parameters["wa"], selected,
        ))
    end

    input_limits = get_minmax_in(frame, INPUT_COLUMNS)
    _, output_array = extract_input_output_df(frame; input_columns=INPUT_COLUMNS)
    output_limits = get_minmax_out(output_array)
    maximin_df!(frame, input_limits, output_limits; input_columns=INPUT_COLUMNS)
    x_train, y_train, x_validation, y_validation, train_indices, validation_indices = getdata(
        frame; test_fraction=0.2, seed=arguments["split-seed"],
        input_columns=INPUT_COLUMNS, return_indices=true,
    )
    network_dictionary = Dict{String,Any}(
        "n_input_features" => length(INPUT_COLUMNS),
        "n_output_features" => size(y_train, 1),
        "n_hidden_layers" => 5,
        "layers" => Dict("layer_$i" => Dict("n_neurons" => 64, "activation_function" => "tanh") for i in 1:5),
        "emulator_description" => Dict(
            "source" => "CLASS + PyBird",
            "cosmology" => "Mnu-w0-waCDM", "component" => component,
            "multipole" => multipole, "preprocessing" => arguments["preprocessing"],
        ),
    )
    network = AbstractCosmologicalEmulators._get_nn_simplechains(network_dictionary)
    output_directory = joinpath(abspath(arguments["path-output"]), string(multipole), component)
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "k.npy"), k)
    npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
    npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
    npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
    npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
    open(joinpath(output_directory, "nn_setup.json"), "w") do stream
        JSON3.write(stream, network_dictionary)
    end
    template = component == "loop" ? "postprocessing_loop" : "postprocessing"
    cp(joinpath(@__DIR__, "$template.jl"), joinpath(output_directory, "postprocessing.jl"); force=true)
    cp(joinpath(@__DIR__, "$template.py"), joinpath(output_directory, "postprocessing.py"); force=true)
    config = SimpleChainsTrainingConfig(
        learning_rates=[1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7],
        sessions_per_rate=arguments["sessions-per-rate"],
        steps_per_session=arguments["steps-per-session"],
        batch_size=arguments["batch-size"], initialization_seed=arguments["initialization-seed"],
    )
    callback = progress -> println(
        "steps=$(progress.total_steps) train=$(progress.training_loss) " *
        "validation=$(progress.validation_loss) best=$(progress.best_validation_loss)",
    )
    result = train_simplechains(
        network, x_train, y_train, x_validation, y_validation;
        config, callback,
        checkpoint_callback=(parameters, progress) ->
            save_training_checkpoint(output_directory, parameters, progress),
    )
    save_training_result(output_directory, result; metadata=Dict(
        "component" => component, "multipole" => multipole,
        "preprocessing" => arguments["preprocessing"], "n_loaded" => n_samples,
        "n_train" => length(train_indices), "n_validation" => length(validation_indices),
        "split_seed" => arguments["split-seed"],
    ))
    println("Best validation loss: $(result.best_validation_loss)")
end

main()
