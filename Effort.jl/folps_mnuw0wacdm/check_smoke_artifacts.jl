using AbstractCosmologicalEmulators
using Effort
using EmulatorsTrainer
using JSON3
using LinearAlgebra
using NPZ
using Statistics
using Test

const ROOT = @__DIR__
const DATASET_PATH = get(ENV, "FOLPS_DATASET", joinpath(ROOT, "data", "smoke_50", "dataset.h5"))
const ARTIFACT_ROOT = get(ENV, "FOLPS_ARTIFACT_ROOT", joinpath(ROOT, "artifacts", "smoke_50"))
const DATASET = load_hdf5_dataset(DATASET_PATH)
const BIAS_COMBINATION = include(joinpath(ROOT, "biascombination.jl"))

function growth(parameters)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=parameters["ln10As"], nₛ=parameters["ns"], h=parameters["H0"] / 100,
        ωb=parameters["ombh2"], ωc=parameters["omch2"], mν=parameters["Mnu"],
        w0=parameters["w0"], wa=parameters["wa"])
    return Effort.D_f_z(parameters["z"], cosmology)
end

function nuisance(f0)
    b1 = 1.645
    return [b1, -0.46, -4 / 7 * (b1 - 1), 32 / 315 * (b1 - 1),
        3.0, -28.9, 2.0, 0.2, 0.08, -8.1, 4719.7358, f0]
end

emulators = Dict(ell => Effort.load_multipole_emulator(
    joinpath(ARTIFACT_ROOT, string(ell)) * "/"; emu=SimpleChainsEmulator) for ell in (0, 2, 4))
validation_indices = Int.(npzread(joinpath(ARTIFACT_ROOT, "0", "11", "validation_indices.npy"))) .+ 1
for ell in (0, 2, 4), component in ("11", "loop", "ct")
    indices = Int.(npzread(joinpath(ARTIFACT_ROOT, string(ell), component, "validation_indices.npy"))) .+ 1
    indices == validation_indices || error("Validation splits differ for ell=$ell component=$component")
end

errors = Dict(ell => Float64[] for ell in (0, 2, 4))
@testset "Folps EFT multipole artifacts" begin
    for sample_index in validation_indices
        input = vec(DATASET.parameters[sample_index, :])
        parameters = Dict(DATASET.parameter_names[j] => input[j] for j in eachindex(input))
        D, f0 = growth(parameters)
        biases = nuisance(f0)
        coefficients = BIAS_COMBINATION(biases)
        for ell in (0, 2, 4)
            prediction = Base.invokelatest(Effort.get_Pℓ, input, D, biases, emulators[ell])
            stoch = ell == 0 ? hcat(ones(59), DATASET.axes[:k] .^ 2 ./ 3) :
                ell == 2 ? hcat(zeros(59), 2 .* DATASET.axes[:k] .^ 2 ./ 3) : zeros(59, 2)
            reference = hcat(DATASET.observables[Symbol("pk_$ell")][sample_index, :, :], stoch) * coefficients
            @test size(prediction) == (59,)
            @test all(isfinite, prediction)
            push!(errors[ell], norm(prediction - reference) / norm(reference))
        end
    end
    for ell in (0, 2, 4), component in ("11", "loop", "ct")
        metadata = JSON3.read(read(joinpath(ARTIFACT_ROOT, string(ell), component, "training_metadata.json"), String))
        @test metadata["n_train"] == size(DATASET.parameters, 1) - length(validation_indices)
        @test metadata["n_validation"] == length(validation_indices)
        @test isfinite(metadata["best_validation_loss"])
    end
end

for ell in (0, 2, 4)
    println("ell=$ell relative L2: median=$(median(errors[ell])) max=$(maximum(errors[ell]))")
end
open(joinpath(ARTIFACT_ROOT, "end_to_end_validation.json"), "w") do stream
    JSON3.write(stream, Dict(
        "validation_indices" => validation_indices,
        "relative_l2" => Dict(string(ell) => errors[ell] for ell in (0, 2, 4)),
        "median" => Dict(string(ell) => median(errors[ell]) for ell in (0, 2, 4)),
        "maximum" => Dict(string(ell) => maximum(errors[ell]) for ell in (0, 2, 4)),
        "dataset" => DATASET_PATH,
        "sample_count" => size(DATASET.parameters, 1),
        "note" => "Short training diagnostic; not a final precision benchmark"))
end
