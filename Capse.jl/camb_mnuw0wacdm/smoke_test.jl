using AbstractCosmologicalEmulators
using Capse
using EmulatorsTrainer

const PROJECT_DIRECTORY = @__DIR__
const DATA_DIRECTORY = joinpath(@__DIR__, "data", "smoke_50")
const ARTIFACT_ROOT = joinpath(@__DIR__, "artifacts", "smoke_50")
const ARTIFACT_DIRECTORY = joinpath(ARTIFACT_ROOT, "TT")

function main()
    isdir(DATA_DIRECTORY) && rm(DATA_DIRECTORY; recursive=true)
    isdir(ARTIFACT_ROOT) && rm(ARTIFACT_ROOT; recursive=true)

    julia = Base.julia_cmd()
    run(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "data_generation.jl")) 50 $DATA_DIRECTORY --force`)

    environment = copy(ENV)
    environment["CAPSE_STEPS_PER_SESSION"] = "100"
    environment["CAPSE_SESSIONS_PER_RATE"] = "1"
    environment["CAPSE_BATCH_SIZE"] = "32"
    run(setenv(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "train.jl")) TT $(joinpath(DATA_DIRECTORY, "dataset.h5")) $ARTIFACT_ROOT`, environment))

    emulator = Capse.load_emulator(ARTIFACT_DIRECTORY * "/"; emu=SimpleChainsEmulator)
    dataset = EmulatorsTrainer.load_hdf5_dataset(joinpath(DATA_DIRECTORY, "dataset.h5"))
    input = vec(dataset.parameters[1, :])
    prediction = Base.invokelatest(Capse.get_Cℓ, input, emulator)
    length(prediction) == 8999 || error("Unexpected TT prediction length: $(length(prediction))")
    all(isfinite, prediction) || error("TT prediction contains NaN or Inf")
    println("CAMB Mnu-w0-wa smoke test passed: 50 samples, finite dense TT prediction length 8999")
end

main()
