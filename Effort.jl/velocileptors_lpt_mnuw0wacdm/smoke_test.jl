using AbstractCosmologicalEmulators
using Effort
using EmulatorsTrainer
using JSON3

const PROJECT_DIRECTORY = @__DIR__
const DATA_DIRECTORY = joinpath(@__DIR__, "data", "smoke_50")
const ARTIFACT_ROOT = joinpath(@__DIR__, "artifacts", "smoke_50")
const ARTIFACT_DIRECTORY = joinpath(ARTIFACT_ROOT, "0", "loop")

function main()
    julia = Base.julia_cmd()
    isdir(DATA_DIRECTORY) && rm(DATA_DIRECTORY; recursive=true)
    isdir(ARTIFACT_ROOT) && rm(ARTIFACT_ROOT; recursive=true)
    run(`$julia --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "data_generation_local.jl")) --samples 50 --output $DATA_DIRECTORY`)
    run(`$julia -t 8 --project=$PROJECT_DIRECTORY $(joinpath(@__DIR__, "trainer.jl")) --component loop --multipole 0 --path-input $(joinpath(DATA_DIRECTORY, "dataset.h5")) --path-output $ARTIFACT_ROOT --steps-per-session 100 --sessions-per-rate 1 --batch-size 32`)

    emulator = Effort.load_component_emulator(
        ARTIFACT_DIRECTORY * "/";
        emu=SimpleChainsEmulator,
        postprocessing_file="postprocessing.jl",
    )
    dataset = EmulatorsTrainer.load_hdf5_dataset(joinpath(DATA_DIRECTORY, "dataset.h5"))
    input = vec(dataset.parameters[1, :])
    prediction = Base.invokelatest(Effort.get_component, input, 0.8, emulator)
    size(prediction) == (59, 9) || error("Unexpected prediction shape: $(size(prediction))")
    all(isfinite, prediction) || error("Smoke-test prediction contains NaN or Inf")

    metadata = JSON3.read(
        read(joinpath(ARTIFACT_DIRECTORY, "training_metadata.json"), String),
    )
    isfinite(metadata["best_validation_loss"]) || error("Validation loss is not finite")
    metadata["n_train"] == 40 || error("Expected 40 training samples")
    metadata["n_validation"] == 10 || error("Expected 10 validation samples")
    println(
        "Smoke test passed: 50 samples, finite loss $(metadata["best_validation_loss"]), " *
        "prediction shape $(size(prediction))",
    )
end

main()
