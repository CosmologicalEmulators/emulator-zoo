using JSON3

const PROJECT = @__DIR__
const DATA = joinpath(PROJECT, "data", "local_500")
const JULIA = Base.julia_cmd()

if !isfile(joinpath(DATA, "dataset.h5"))
    run(`$JULIA --project=$PROJECT $(joinpath(PROJECT, "data_generation_local.jl"))
        --samples 500 --processes 4 --output $DATA`)
end

for ad_backend in ("zygote", "reactant")
    artifacts = joinpath(PROJECT, "artifacts", "local_500_lux_$(ad_backend)")
    for basis in ("ln10As", "sigma8")
        cmd = `$JULIA --project=$PROJECT $(joinpath(PROJECT, "trainer_lux.jl"))
            --basis $basis
            --path-input $(joinpath(DATA, "dataset.h5"))
            --path-output $artifacts
            --steps-per-session 50
            --sessions-per-rate 1
            --batch-size 32
            --ad-backend $ad_backend`
        if ad_backend == "reactant"
            cmd = `$cmd --reactant-backend cpu`
        end
        run(cmd)
        metadata = JSON3.read(read(joinpath(artifacts, basis, "training_metadata.json"), String))
        isfinite(metadata["best_validation_loss"]) || error("Non-finite Lux validation loss for $basis ($ad_backend)")
    end
end

println("ACE Lux 500-sample smoke test passed for both bases and both backends (Zygote + Reactant CPU)")
