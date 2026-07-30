using BenchmarkTools

include(joinpath(@__DIR__, "generation.jl"))
using .ACEClassMnuW0WaGeneration

const PARAMETERS = Dict(
    "z" => 0.8, "ln10As" => 3.044, "ns" => 0.965, "H0" => 67.4,
    "ombh2" => 0.0224, "omch2" => 0.12, "Mν" => 0.06, "w0" => -1.0, "wa" => 0.0,
)
const BACKEND = ACEClassMnuW0WaGeneration.initialize_backend()
result = ACEClassMnuW0WaGeneration.compute_observables(PARAMETERS, BACKEND)
all(values -> all(isfinite, values), result) || error("Benchmark result is not finite")
trial = @benchmark ACEClassMnuW0WaGeneration.compute_observables($PARAMETERS, $BACKEND) samples=5 evals=1
println("median_seconds=$(median(trial).time / 1e9)")
println("minimum_seconds=$(minimum(trial).time / 1e9)")

