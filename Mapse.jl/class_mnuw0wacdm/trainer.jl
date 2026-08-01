using AbstractCosmologicalEmulators
using ArgParse
using EmulatorsTrainer
using JSON3
using Mapse
using NPZ
using Random

const FULL_PARAMETER_NAMES = ["ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
const NETWORK_PARAMETER_NAMES = ["z", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]

function minmax_rows(array::AbstractMatrix)
    return hcat(vec(minimum(array; dims=2)), vec(maximum(array; dims=2)))
end

function normalize_rows(array::AbstractMatrix, limits::AbstractMatrix)
    widths = limits[:, 2] .- limits[:, 1]
    all(>(0), widths) || error("Cannot normalize a constant feature")
    return (array .- reshape(limits[:, 1], :, 1)) ./ reshape(widths, :, 1)
end

settings = ArgParseSettings()
@add_arg_table settings begin
    "--spectrum"; default="Pmm"
    "--path-input", "-i"; required=true
    "--path-output", "-o"; required=true
    "--pca-components"; arg_type=Int; default=17
    "--split-seed"; arg_type=Int; default=20260764
    "--initialization-seed"; arg_type=Int; default=20260765
    "--steps-per-session"; arg_type=Int; default=2000
    "--sessions-per-rate"; arg_type=Int; default=10
    "--batch-size"; arg_type=Int; default=256
end
arguments = parse_args(settings)
spectrum = arguments["spectrum"]
spectrum in ("Pmm", "Pcb") || error("--spectrum must be Pmm or Pcb")
observable_name = spectrum == "Pmm" ? :Pk_lin_mm : :Pk_lin_cb
artifact_name = String(observable_name)

dataset = load_hdf5_dataset(abspath(arguments["path-input"]))
all(dataset.valid) || error("HDF5 dataset contains invalid samples")
parameters = dataset.parameters
parameter_names = dataset.parameter_names
spectra = dataset.observables[observable_name]
k = dataset.axes[:k]
size(spectra) == (size(parameters, 1), length(k)) || error("Spectrum and k-grid shapes disagree")

indices = Dict(name => only(findall(==(name), parameter_names)) for name in parameter_names)
n_samples = size(parameters, 1)
network_inputs = Matrix{Float64}(undef, length(NETWORK_PARAMETER_NAMES), n_samples)
targets = Matrix{Float64}(undef, length(k), n_samples)
for sample_index in 1:n_samples
    p = Dict(name => parameters[sample_index, indices[name]] for name in parameter_names)
    full_parameters = [p[name] for name in FULL_PARAMETER_NAMES]
    z = p["z"]
    h = p["H0"] / 100
    cosmology = Mapse.w0waCDMCosmology(
        ln10Aₛ=p["ln10As"], nₛ=p["ns"], h=h,
        ωb=p["ombh2"], ωc=p["omch2"], mν=p["Mν"], w0=p["w0"], wa=p["wa"],
    )
    D = Mapse.D_z(z, cosmology)
    As = exp(p["ln10As"]) * 1e-10
    primordial = Mapse.primordial_Pk(As, p["ns"], k)
    transfer = Mapse.lcdm_transfer_function(full_parameters, k)
    target = vec(spectra[sample_index, :]) ./ (D^2 .* primordial .* transfer.^2)
    all(isfinite, target) || error("Non-finite transformed target at sample $sample_index")
    targets[:, sample_index] = target
    network_inputs[:, sample_index] = [p[name] for name in NETWORK_PARAMETER_NAMES]
end

n_components = arguments["pca-components"]
1 <= n_components <= min(size(targets)...) || error("Invalid number of PCA components")
pca_mean, pca_basis, coefficients = Mapse.compute_pca(targets, n_components)
input_limits = minmax_rows(network_inputs)
output_limits = minmax_rows(coefficients)
normalized_inputs = normalize_rows(network_inputs, input_limits)
normalized_outputs = normalize_rows(coefficients, output_limits)

Random.seed!(arguments["split-seed"])
permutation = randperm(n_samples)
n_validation = max(1, round(Int, 0.2 * n_samples))
n_validation < n_samples || error("At least two samples are required")
validation_indices = sort(permutation[1:n_validation])
train_indices = sort(permutation[(n_validation + 1):end])
x_train = normalized_inputs[:, train_indices]
y_train = normalized_outputs[:, train_indices]
x_validation = normalized_inputs[:, validation_indices]
y_validation = normalized_outputs[:, validation_indices]

network_dictionary = Dict{String,Any}(
    "n_input_features" => size(x_train, 1),
    "n_output_features" => size(y_train, 1),
    "n_hidden_layers" => 5,
    "layers" => Dict(
        "layer_$index" => Dict("n_neurons" => 64, "activation_function" => "tanh")
        for index in 1:5
    ),
    "preprocessing_name" => "drop_primordial_parameters",
    "postprocessing_name" => "lcdm_transfer_ratio",
    "emulator_description" => Dict(
        "source" => "CLASS linear power spectrum",
        "cosmology" => "Mnu-w0-waCDM",
        "spectrum" => spectrum,
        "units" => "k in Mpc^-1, P(k) in Mpc^3",
        "parameters" => join(FULL_PARAMETER_NAMES, ", "),
    ),
)
network = AbstractCosmologicalEmulators._get_nn_simplechains(network_dictionary)
output_directory = joinpath(abspath(arguments["path-output"]), artifact_name)
mkpath(output_directory)
npzwrite(joinpath(output_directory, "k.npy"), k)
npzwrite(joinpath(output_directory, "inminmax.npy"), input_limits)
npzwrite(joinpath(output_directory, "outminmax.npy"), output_limits)
npzwrite(joinpath(output_directory, "train_indices.npy"), train_indices .- 1)
npzwrite(joinpath(output_directory, "validation_indices.npy"), validation_indices .- 1)
Mapse.save_pca_metadata(output_directory, pca_mean, pca_basis)
open(joinpath(output_directory, "nn_setup.json"), "w") do stream
    JSON3.write(stream, network_dictionary)
end
config = SimpleChainsTrainingConfig(
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
result = train_simplechains(
    network, x_train, y_train, x_validation, y_validation;
    config, callback,
    checkpoint_callback=(parameters, progress) ->
        save_training_checkpoint(output_directory, parameters, progress),
)
save_training_result(output_directory, result; metadata=Dict(
    "spectrum" => spectrum,
    "n_loaded" => n_samples,
    "n_train" => length(train_indices),
    "n_validation" => length(validation_indices),
    "pca_components" => n_components,
    "split_seed" => arguments["split-seed"],
))
println("Best validation loss: $(result.best_validation_loss)")
