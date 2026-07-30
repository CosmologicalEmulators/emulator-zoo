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
    run(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "data_generation_local.jl")) --samples 50 --output $DATA_DIRECTORY`)
    env = copy(ENV)
    env["CAPSE_STEPS_PER_SESSION"] = "100"
    env["CAPSE_SESSIONS_PER_RATE"] = "1"
    env["CAPSE_BATCH_SIZE"] = "32"
    run(setenv(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "trainer.jl")) --spectrum TT -i $(joinpath(DATA_DIRECTORY, "dataset.h5")) -o $ARTIFACT_ROOT`, env))
    emulator = Capse.load_emulator(ARTIFACT_DIRECTORY * "/"; emu=SimpleChainsEmulator)
    dataset = EmulatorsTrainer.load_hdf5_dataset(joinpath(DATA_DIRECTORY, "dataset.h5"))
    input = vec(dataset.parameters[1, :])
    prediction = Base.invokelatest(Capse.get_Cℓ, input, emulator)
    length(prediction) == 2999 || error("Unexpected TT prediction length: $(length(prediction))")
    all(isfinite, prediction) || error("TT prediction contains NaN or Inf")
    println("axiclass smoke test passed: 50 samples, finite TT prediction length 2999")
end

main()
