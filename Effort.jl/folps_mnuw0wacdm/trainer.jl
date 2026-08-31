using AbstractCosmologicalEmulators
using ArgParse
using DataFrames
using Effort
using EmulatorsTrainer
using JSON3
using NPZ
using SimpleChains

const INPUT_COLUMNS = [:z, :ln10As, :ns, :H0, :ombh2, :omch2, :Mnu, :w0, :wa]
const COMPONENTS = ("11", "loop", "ct")

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--component"; required=true
        "--multipole"; arg_type=Int; required=true
        "--path-input"; required=true
        "--path-output"; required=true
        "--steps-per-session"; arg_type=Int; default=1000
        "--sessions-per-rate"; arg_type=Int; default=10
        "--batch-size"; arg_type=Int; default=512
        "--split-seed"; arg_type=Int; default=20260901
        "--initialization-seed"; arg_type=Int; default=20260902
    end
    return parse_args(settings)
end

component_columns(component) = component == "11" ? (1:3) : component == "loop" ? (4:15) : component == "ct" ? (16:21) : error("Unknown component: $component")

function growth_factor(parameters)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=3.0, nₛ=0.96, h=parameters["H0"] / 100,
        ωb=parameters["ombh2"], ωc=parameters["omch2"], mν=parameters["Mnu"],
        w0=parameters["w0"], wa=parameters["wa"],
    )
    return Effort.D_z(parameters["z"], cosmology)
end

amplitude(parameters) = exp(parameters["ln10As"]) * 1.0e-10 * growth_factor(parameters)^2

function preprocess(values, parameters, component)
    scale = amplitude(parameters)
    component == "11" && return vec(values ./ scale)
    component == "loop" && return vec(values ./ scale^2)
    return vcat(vec(values[:, 1:3]) ./ scale, vec(values[:, 4:6]) ./ scale^3)
end

function write_postprocessing(directory, component)
    if component == "11"
        julia_source = "(input, output, D, emulator) -> output .* (exp(input[2]) * 1.0e-10 * D^2)\n"
        python_source = """import jax.numpy as jnp

def postprocessing(input, output, D):
    return output * (jnp.exp(input[..., 1]) * 1.0e-10 * D**2)
"""
    elseif component == "loop"
        julia_source = "(input, output, D, emulator) -> output .* (exp(input[2]) * 1.0e-10 * D^2)^2\n"
        python_source = """import jax.numpy as jnp

def postprocessing(input, output, D):
    return output * (jnp.exp(input[..., 1]) * 1.0e-10 * D**2)**2
"""
    else
        julia_source = """(input, output, D, emulator) -> begin
    scale = exp(input[2]) * 1.0e-10 * D^2
    n = length(output) ÷ 6
    vcat(output[1:(3n)] .* scale, output[(3n + 1):(6n)] .* scale^3)
end
"""
        python_source = """import jax.numpy as jnp

def postprocessing(input, output, D):
    scale = jnp.exp(input[..., 1]) * 1.0e-10 * D**2
    n = output.shape[-1] // 6
    return jnp.concatenate((output[..., :3*n] * scale, output[..., 3*n:] * scale**3), axis=-1)
"""
    end
    write(joinpath(directory, "postprocessing.jl"), julia_source)
    write(joinpath(directory, "postprocessing.py"), python_source)
end

function main()
    arguments = parse_commandline()
    component, multipole = arguments["component"], arguments["multipole"]
    component in COMPONENTS || error("Unknown component: $component")
    multipole in (0, 2, 4) || error("Multipole must be 0, 2, or 4")
    columns = component_columns(component)
    dataset = load_hdf5_dataset(abspath(arguments["path-input"]))
    all(dataset.valid) || error("Dataset contains invalid samples")
    observable = get(dataset.observables, Symbol("pk_$multipole"), nothing)
    observable === nothing && error("Dataset has no pk_$multipole observable")
    size(observable)[2:3] == (59, 21) || error("Unexpected observable shape: $(size(observable))")

    frame = DataFrame(sample_id=String[], z=Float64[], ln10As=Float64[], ns=Float64[],
        H0=Float64[], ombh2=Float64[], omch2=Float64[], Mnu=Float64[], w0=Float64[], wa=Float64[],
        observable=Vector{Float64}[])
    for sample_index in axes(dataset.parameters, 1)
        parameters = Dict(dataset.parameter_names[j] => dataset.parameters[sample_index, j] for j in axes(dataset.parameters, 2))
        target = preprocess(Array(observable[sample_index, :, columns]), parameters, component)
        push!(frame, (sample_id="sample_$(lpad(sample_index, 6, '0'))", z=parameters["z"],
            ln10As=parameters["ln10As"], ns=parameters["ns"], H0=parameters["H0"],
            ombh2=parameters["ombh2"], omch2=parameters["omch2"], Mnu=parameters["Mnu"],
            w0=parameters["w0"], wa=parameters["wa"], observable=target))
    end

    input_limits = get_minmax_in(frame, INPUT_COLUMNS)
    _, output_array = extract_input_output_df(frame; input_columns=INPUT_COLUMNS)
    output_limits = get_minmax_out(output_array)
    maximin_df!(frame, input_limits, output_limits; input_columns=INPUT_COLUMNS)
    x_train, y_train, x_validation, y_validation, train_indices, validation_indices = getdata(
        frame; test_fraction=0.2, seed=arguments["split-seed"], input_columns=INPUT_COLUMNS, return_indices=true)
    setup = Dict{String,Any}("n_input_features" => length(INPUT_COLUMNS), "n_output_features" => size(y_train, 1),
        "n_hidden_layers" => 5, "layers" => Dict("layer_$i" => Dict("n_neurons" => 64, "activation_function" => "tanh") for i in 1:5),
        "emulator_description" => Dict("source" => "CLASS + Folps EFT", "component" => component,
            "multipole" => multipole, "runtime_model" => "EFT", "AP_during_generation" => false))
    network = AbstractCosmologicalEmulators._get_nn_simplechains(setup)
    multipole_directory = joinpath(abspath(arguments["path-output"]), string(multipole))
    output_directory = joinpath(multipole_directory, component)
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "k.npy"), vec(dataset.axes[:k]))
    npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
    npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
    npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
    npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
    open(joinpath(output_directory, "nn_setup.json"), "w") do stream; JSON3.write(stream, setup); end
    write_postprocessing(output_directory, component)
    for filename in ("biascombination.jl", "biascombination.py", "jacbiascombination.jl", "jacbiascombination.py")
        cp(joinpath(@__DIR__, filename), joinpath(multipole_directory, filename); force=true)
    end
    cp(joinpath(@__DIR__, "stochmodel_$multipole.jl"), joinpath(multipole_directory, "stochmodel.jl"); force=true)
    cp(joinpath(@__DIR__, "stochmodel_$multipole.py"), joinpath(multipole_directory, "stochmodel.py"); force=true)
    config = SimpleChainsTrainingConfig(learning_rates=[1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7],
        sessions_per_rate=arguments["sessions-per-rate"], steps_per_session=arguments["steps-per-session"],
        batch_size=arguments["batch-size"], initialization_seed=arguments["initialization-seed"])
    callback = progress -> begin
        println("ell=$multipole component=$component steps=$(progress.total_steps) train=$(progress.training_loss) validation=$(progress.validation_loss) best=$(progress.best_validation_loss)")
        flush(stdout)
    end
    result = train_simplechains(network, x_train, y_train, x_validation, y_validation; config, callback,
        checkpoint_callback=(parameters, progress) -> save_training_checkpoint(output_directory, parameters, progress))
    save_training_result(output_directory, result; metadata=Dict("component" => component, "multipole" => multipole,
        "n_loaded" => nrow(frame), "n_train" => length(train_indices), "n_validation" => length(validation_indices),
        "train_sample_ids" => frame.sample_id[train_indices], "validation_sample_ids" => frame.sample_id[validation_indices]))
end

main()
