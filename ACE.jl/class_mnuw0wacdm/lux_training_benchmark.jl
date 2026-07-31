using JSON3

const PROJECT = @__DIR__
const DATASET = joinpath(PROJECT, "data", "local_500", "dataset.h5")
const OUTPUT = joinpath(PROJECT, "artifacts", "lux_training_benchmark_100x10")
const JULIA = Base.julia_cmd()
const BASIS = isempty(ARGS) ? "ln10As" : first(ARGS)
BASIS in ("ln10As", "sigma8") || error("Basis must be ln10As or sigma8")

isfile(DATASET) || error("Missing local dataset: $DATASET")

timings = Dict{String,Float64}()
for backend in ("zygote", "reactant")
    artifact_root = joinpath(OUTPUT, backend)
    command = `$JULIA --project=$PROJECT $(joinpath(PROJECT, "trainer_lux.jl"))
        --basis $BASIS
        --path-input $DATASET
        --path-output $artifact_root
        --steps-per-session 100
        --sessions-per-rate 10
        --batch-size 32
        --ad-backend $backend
        --warmup-steps $(backend == "reactant" ? 1 : 0)`
    if backend == "reactant"
        command = `$command --reactant-backend cpu`
    end
    run(command)
    metadata_path = joinpath(artifact_root, BASIS, "training_metadata.json")
    metadata = JSON3.read(read(metadata_path, String))
    timings[backend] = Float64(metadata["training_seconds"])
    println("$backend: $(timings[backend]) seconds (warmup excluded)")
end

zygote_seconds = timings["zygote"]
reactant_seconds = timings["reactant"]
println("Reactant/Zygote post-warmup runtime ratio: $(reactant_seconds / zygote_seconds)")
println("Faster backend after compilation: ", reactant_seconds < zygote_seconds ? "Reactant" : "Zygote")
