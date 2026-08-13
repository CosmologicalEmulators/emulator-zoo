using AbstractCosmologicalEmulators
using EmulatorsTrainer
using JSON3
using Mapse

const PROJECT = @__DIR__
const DATA = joinpath(PROJECT, "data", "smoke_50")
const ARTIFACTS = joinpath(PROJECT, "artifacts", "smoke_50")
isdir(DATA) && rm(DATA; recursive=true)
isdir(ARTIFACTS) && rm(ARTIFACTS; recursive=true)
julia = Base.julia_cmd()
run(`$julia --project=$PROJECT $(joinpath(PROJECT, "data_generation_local.jl")) --samples 50 --processes 2 --output $DATA`)
for spectrum in ("Pmm", "Pcb")
    run(`$julia --project=$PROJECT $(joinpath(PROJECT, "trainer.jl")) --spectrum $spectrum -i $(joinpath(DATA, "dataset.h5")) -o $ARTIFACTS --steps-per-session 100 --sessions-per-rate 1 --batch-size 32`)
end

dataset = load_hdf5_dataset(joinpath(DATA, "dataset.h5"))
p = Dict(dataset.parameter_names[j] => dataset.parameters[1, j] for j in axes(dataset.parameters, 2))
input = [p[name] for name in ("ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa")]
h = p["H0"] / 100
cosmology = Mapse.w0waCDMCosmology(
    ln10Aₛ=p["ln10As"], nₛ=p["ns"], h=h,
    ωb=p["ombh2"], ωc=p["omch2"], mν=p["Mν"], w0=p["w0"], wa=p["wa"],
)
D = Mapse.D_z(p["z"], cosmology)
for artifact in ("Pk_lin_mm", "Pk_lin_cb")
    directory = joinpath(ARTIFACTS, artifact)
    emulator = Mapse.load_emulator(directory; emu=SimpleChainsEmulator)
    prediction = Mapse.get_Pk(input, p["z"], D, emulator)
    length(prediction) == 300 || error("Unexpected $artifact prediction length")
    all(isfinite, prediction) || error("Non-finite $artifact prediction")
    all(>(0), prediction) || error("Non-positive $artifact prediction")
    metadata = JSON3.read(read(joinpath(directory, "training_metadata.json"), String))
    metadata["n_train"] == 40 || error("Expected 40 training samples")
    metadata["n_validation"] == 10 || error("Expected 10 validation samples")
    isfinite(metadata["best_validation_loss"]) || error("Non-finite validation loss")
end
println("Mapse CLASS Pmm/Pcb smoke test passed")
