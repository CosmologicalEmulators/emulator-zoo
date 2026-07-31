using AbstractCosmologicalEmulators
using ADTypes
using ArgParse
using DataFrames
using EmulatorsTrainer
using JSON3
using Lux
using NPZ

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]

settings = ArgParseSettings()
@add_arg_table settings begin
    "--basis"; default="ln10As"
    "--path-input", "-i"; required=true
    "--path-output", "-o"; required=true
    "--split-seed"; arg_type=Int; default=20260761
    "--initialization-seed"; arg_type=Int; default=20260762
    "--steps-per-session"; arg_type=Int; default=4000
    "--sessions-per-rate"; arg_type=Int; default=40
    "--batch-size"; arg_type=Int; default=256
    "--ad-backend"; default="zygote"
    "--reactant-backend"; default="cpu"
    "--warmup-steps"; arg_type=Int; default=0
end
arguments = parse_args(settings)
basis = arguments["basis"]
basis in ("ln10As", "sigma8") || error("--basis must be ln10As or sigma8")
ad_backend_name = arguments["ad-backend"]
ad_backend_name in ("zygote", "reactant") || error("--ad-backend must be zygote or reactant")
arguments["warmup-steps"] >= 0 || error("--warmup-steps must be non-negative")

dataset = load_hdf5_dataset(abspath(arguments["path-input"]))
all(dataset.valid) || error("HDF5 dataset contains invalid samples")
parameters = dataset.parameters
parameter_names = dataset.parameter_names
ln10As_outputs = dataset.observables[:result_ln10As_basis]
sigma8_outputs = dataset.observables[:result_sigma8_basis]

frame = if basis == "ln10As"
    DataFrame(z=Float64[], ln10As=Float64[], ns=Float64[], H0=Float64[],
        ombh2=Float64[], omch2=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[],
        observable=Vector{Float64}[])
else
    DataFrame(z=Float64[], sigma8=Float64[], ns=Float64[], H0=Float64[],
        ombh2=Float64[], omch2=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[],
        observable=Vector{Float64}[])
end

for sample_index in axes(parameters, 1)
    p = Dict(parameter_names[j] => parameters[sample_index, j] for j in axes(parameters, 2))
    if basis == "ln10As"
        push!(frame, (p["z"], p["ln10As"], p["ns"], p["H0"], p["ombh2"], p["omch2"],
            p["Mν"], p["w0"], p["wa"], vec(ln10As_outputs[sample_index, :])))
    else
        sigma8 = ln10As_outputs[sample_index, 1]
        push!(frame, (p["z"], sigma8, p["ns"], p["H0"], p["ombh2"], p["omch2"],
            p["Mν"], p["w0"], p["wa"], vec(sigma8_outputs[sample_index, :])))
    end
end

input_columns = basis == "ln10As" ?
    [:z, :ln10As, :ns, :H0, :ombh2, :omch2, :Mν, :w0, :wa] :
    [:z, :sigma8, :ns, :H0, :ombh2, :omch2, :Mν, :w0, :wa]
input_limits = get_minmax_in(frame, input_columns)
_, output_array = extract_input_output_df(frame; input_columns)
output_limits = get_minmax_out(output_array)
maximin_df!(frame, input_limits, output_limits; input_columns)
x_train, y_train, x_validation, y_validation, train_indices, validation_indices = getdata(
    frame; test_fraction=0.2, seed=arguments["split-seed"], input_columns, return_indices=true,
)

network_dictionary = Dict{String,Any}(
    "n_input_features" => 9,
    "n_output_features" => 7,
    "n_hidden_layers" => 5,
    "layers" => Dict(
        "layer_$index" => Dict("n_neurons" => 64, "activation_function" => "tanh")
        for index in 1:5
    ),
    "emulator_description" => Dict(
        "source" => "CLASS + AbstractCosmologicalEmulators BackgroundCosmologyExt growth ODE",
        "cosmology" => "Mnu-w0-waCDM",
        "basis" => basis,
        "parameters" => join(string.(input_columns), ", "),
        "training_backend" => "Lux.jl",
        "postprocessing_name" => "identity",
    ),
    "postprocessing_name" => "identity",
)
network = AbstractCosmologicalEmulators._get_nn_lux(network_dictionary)
output_directory = joinpath(abspath(arguments["path-output"]), basis)
mkpath(output_directory)
npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
open(joinpath(output_directory, "nn_setup.json"), "w") do stream
    JSON3.write(stream, network_dictionary)
end
write(joinpath(output_directory, "postprocessing.jl"), "(params, output, auxiliary_params, emu) -> output")
config = LuxTrainingConfig(
    learning_rates=[1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7],
    sessions_per_rate=arguments["sessions-per-rate"],
    steps_per_session=arguments["steps-per-session"],
    batch_size=arguments["batch-size"],
    initialization_seed=arguments["initialization-seed"],
)
callback = progress -> println(
    "steps=$(progress.total_steps) train=$(progress.training_loss) " *
    "validation=$(progress.validation_loss) best=$(progress.best_validation_loss)",
)
training_ad_backend, training_device = if ad_backend_name == "reactant"
        @eval using Enzyme
        @eval using Reactant
        Reactant.set_default_backend(arguments["reactant-backend"])
        (AutoEnzyme(), Lux.reactant_device(; force=true))
    else
        (AutoZygote(), identity)
    end

warmup_seconds = 0.0
if arguments["warmup-steps"] > 0
    warmup_config = LuxTrainingConfig(
        learning_rates=[config.learning_rates[1]],
        sessions_per_rate=1,
        steps_per_session=arguments["warmup-steps"],
        batch_size=config.batch_size,
        initialization_seed=config.initialization_seed,
    )
    warmup_start = time()
    train_lux(
        network,
        x_train,
        y_train,
        x_validation,
        y_validation;
        config=warmup_config,
        ad_backend=training_ad_backend,
        device=training_device,
    )
    warmup_seconds = time() - warmup_start
end

training_start = time()
result = train_lux(
    network,
    x_train,
    y_train,
    x_validation,
    y_validation;
    config,
    ad_backend=training_ad_backend,
    device=training_device,
    callback,
)
training_seconds = time() - training_start

save_training_result(output_directory, result; metadata=Dict(
    "basis" => basis,
    "n_loaded" => size(frame, 1),
    "n_train" => length(train_indices),
    "n_validation" => length(validation_indices),
    "split_seed" => arguments["split-seed"],
    "architecture" => network_dictionary,
    "ad_backend" => ad_backend_name,
    "reactant_backend" => ad_backend_name == "reactant" ? arguments["reactant-backend"] : nothing,
    "warmup_steps" => arguments["warmup-steps"],
    "warmup_seconds" => warmup_seconds,
    "training_seconds" => training_seconds,
))
println("Best validation loss: $(result.best_validation_loss)")
println("Training seconds (warmup excluded): $training_seconds")
