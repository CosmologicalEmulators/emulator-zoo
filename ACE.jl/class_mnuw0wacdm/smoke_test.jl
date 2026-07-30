using EmulatorsTrainer
using JSON3

const PROJECT = @__DIR__
const DATA = joinpath(PROJECT, "data", "smoke_50")
const ARTIFACTS = joinpath(PROJECT, "artifacts", "smoke_50")

isdir(DATA) && rm(DATA; recursive=true)
isdir(ARTIFACTS) && rm(ARTIFACTS; recursive=true)
julia = Base.julia_cmd()
run(`$julia --project=$PROJECT $(joinpath(PROJECT, "data_generation_local.jl")) --samples 50 --processes 2 --output $DATA`)
for basis in ("ln10As", "sigma8")
    run(`$julia --project=$PROJECT $(joinpath(PROJECT, "trainer.jl")) --basis $basis -i $(joinpath(DATA, "dataset.h5")) -o $ARTIFACTS --steps-per-session 100 --sessions-per-rate 1 --batch-size 32`)
    metadata = JSON3.read(read(joinpath(ARTIFACTS, basis, "training_metadata.json"), String))
    metadata["n_train"] == 40 || error("Expected 40 training samples for $basis")
    metadata["n_validation"] == 10 || error("Expected 10 validation samples for $basis")
    isfinite(metadata["best_validation_loss"]) || error("Non-finite validation loss for $basis")
end
dataset = load_hdf5_dataset(joinpath(DATA, "dataset.h5"))
size(dataset.parameters) == (50, 9) || error("Unexpected parameter shape")
size(dataset.observables[:result_ln10As_basis]) == (50, 7) || error("Unexpected output shape")
println("ACE CLASS Mnu-w0-wa smoke test passed for both bases")

