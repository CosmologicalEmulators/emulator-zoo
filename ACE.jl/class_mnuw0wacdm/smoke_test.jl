using EmulatorsTrainer
using JSON3

const PROJECT = @__DIR__
const DATA = joinpath(PROJECT, "data", "smoke_50")
const ARTIFACTS = joinpath(PROJECT, "artifacts", "smoke_50")

isdir(DATA) && rm(DATA; recursive=true)
isdir(ARTIFACTS) && rm(ARTIFACTS; recursive=true)
julia = Base.julia_cmd()
run(`$julia --project=$PROJECT $(joinpath(PROJECT, "data_generation_local.jl")) --samples 50 --processes 2 --output $DATA`)
dataset = load_hdf5_dataset(joinpath(DATA, "dataset.h5"))
n_samples = size(dataset.parameters, 1)
n_validation = round(Int, 0.2 * n_samples)
n_training = n_samples - n_validation
for basis in ("ln10As", "sigma8")
    run(`$julia --project=$PROJECT $(joinpath(PROJECT, "trainer.jl")) --basis $basis -i $(joinpath(DATA, "dataset.h5")) -o $ARTIFACTS --steps-per-session 100 --sessions-per-rate 1 --batch-size 32`)
    metadata = JSON3.read(read(joinpath(ARTIFACTS, basis, "training_metadata.json"), String))
    metadata["n_train"] == n_training || error("Unexpected training sample count for $basis")
    metadata["n_validation"] == n_validation || error("Unexpected validation sample count for $basis")
    isfinite(metadata["best_validation_loss"]) || error("Non-finite validation loss for $basis")
end
size(dataset.parameters) == (n_samples, 9) || error("Unexpected parameter shape")
size(dataset.observables[:result_ln10As_basis]) == (n_samples, 7) || error("Unexpected output shape")
println("ACE CLASS Mnu-w0-wa smoke test passed for both bases")
