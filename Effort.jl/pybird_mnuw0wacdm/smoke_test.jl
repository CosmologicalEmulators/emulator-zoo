using EmulatorsTrainer

const PROJECT = @__DIR__
const DATA = joinpath(PROJECT, "data", "smoke_50")
const ARTIFACTS = joinpath(PROJECT, "artifacts", "smoke_50")
isdir(DATA) && rm(DATA; recursive=true)
isdir(ARTIFACTS) && rm(ARTIFACTS; recursive=true)
julia = Base.julia_cmd()
run(`$julia --project=$PROJECT $(joinpath(PROJECT, "data_generation_local.jl")) --samples 50 --processes 2 --output $DATA`)
run(`$julia --project=$PROJECT $(joinpath(PROJECT, "trainer.jl")) --component loop --multipole 0 --preprocessing AsDzprec --path-input $(joinpath(DATA, "dataset.h5")) --path-output $ARTIFACTS --steps-per-session 100 --sessions-per-rate 1 --batch-size 32`)
dataset = load_hdf5_dataset(joinpath(DATA, "dataset.h5"))
size(dataset.parameters, 1) >= 2 || error("Too few valid samples")
size(dataset.observables[:Ploopl], 2) == 3 || error("Unexpected multipole dimension")
println("PyBird Mnu-w0-wa smoke test passed")
