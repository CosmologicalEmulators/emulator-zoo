using AbstractCosmologicalEmulators
using Capse
using EmulatorsTrainer
using JSON3
using NPZ

const PROJECT_DIRECTORY = @__DIR__
const DATA_DIRECTORY = joinpath(@__DIR__, "data", "smoke_50")
const ARTIFACT_ROOT = joinpath(@__DIR__, "artifacts", "smoke_50")
const ARTIFACT_DIRECTORY = joinpath(ARTIFACT_ROOT, "EE")

function main()
    isdir(DATA_DIRECTORY) && rm(DATA_DIRECTORY; recursive=true)
    isdir(ARTIFACT_ROOT) && rm(ARTIFACT_ROOT; recursive=true)

    julia = Base.julia_cmd()
    run(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "data_generation.jl")) 50 $DATA_DIRECTORY --force`)

    environment = copy(ENV)
    environment["CAPSE_STEPS_PER_SESSION"] = "100"
    environment["CAPSE_SESSIONS_PER_RATE"] = "1"
    environment["CAPSE_BATCH_SIZE"] = "32"
    run(setenv(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "train.jl")) EE_LOG $(joinpath(DATA_DIRECTORY, "dataset.h5")) $ARTIFACT_ROOT`, environment))

    emulator = Capse.load_emulator(ARTIFACT_DIRECTORY * "/"; emu=SimpleChainsEmulator)
    training_ell = vec(npzread(joinpath(ARTIFACT_DIRECTORY, "l.npy")))
    metadata = JSON3.read(
        read(joinpath(ARTIFACT_DIRECTORY, "training_metadata.json"), String),
    )
    length(training_ell) == 274 || error("Unexpected EE training grid length")
    training_ell[1:19] == collect(2.0:20.0) || error("EE low-ell grid is not dense")
    metadata["dense_lowell_max"] == 20 || error("Hybrid EE metadata is missing")
    dataset = EmulatorsTrainer.load_hdf5_dataset(joinpath(DATA_DIRECTORY, "dataset.h5"))
    input = vec(dataset.parameters[1, :])
    prediction = Base.invokelatest(Capse.get_Cℓ, input, emulator)
    length(prediction) == 9499 || error("Unexpected EE prediction length: $(length(prediction))")
    all(isfinite, prediction) || error("EE prediction contains NaN or Inf")
    println("CAMB Mnu-OmegaK-LambdaCDM smoke test passed: 50 samples, hybrid EE grid, finite dense prediction length 9499")
end

main()
