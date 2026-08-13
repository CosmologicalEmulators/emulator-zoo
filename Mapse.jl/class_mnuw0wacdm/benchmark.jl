using BenchmarkTools

include(joinpath(@__DIR__, "generation.jl"))
using .MapseClassMnuW0WaGeneration

const PARAMETERS = Dict(
    "z" => 0.8, "H0" => 67.4, "ombh2" => 0.0224,
    "omch2" => 0.12, "Mν" => 0.06, "w0" => -1.0, "wa" => 0.0,
)
const BACKEND = MapseClassMnuW0WaGeneration.initialize_backend()
result = MapseClassMnuW0WaGeneration.compute_observables(PARAMETERS, BACKEND)
all(values -> all(isfinite, values), result) || error("Benchmark result is not finite")
trial = @benchmark MapseClassMnuW0WaGeneration.compute_observables($PARAMETERS, $BACKEND) samples=5 evals=1
println("median_seconds=$(median(trial).time / 1e9)")
println("minimum_seconds=$(minimum(trial).time / 1e9)")
