using DelimitedFiles
using Test

include(joinpath(@__DIR__, "..", "generation.jl"))
using .MapseClassMnuW0WaGeneration

input_line = only(filter(
    line -> !startswith(line, "#") && !isempty(strip(line)),
    readlines(joinpath(@__DIR__, "reference_inputs.txt")),
))
values = parse.(Float64, split(input_line))
parameters = Dict(MapseClassMnuW0WaGeneration.PARAMETER_NAMES .=> values)
reference = readdlm(joinpath(@__DIR__, "reference_outputs.txt"), comments=true)
result = MapseClassMnuW0WaGeneration.compute_observables(
    parameters, MapseClassMnuW0WaGeneration.initialize_backend(),
)

@testset "Mapse CLASS Pmm/Pcb reference" begin
    @test MapseClassMnuW0WaGeneration.K_GRID == reference[:, 1]
    for (name, column) in ((:Pk_lin_mm, 2), (:Pk_lin_cb, 3))
        actual = getproperty(result, name)
        expected = reference[:, column]
        difference = abs.(actual .- expected)
        relative = difference ./ max.(abs.(expected), eps(Float64))
        println("$name: max_abs=$(maximum(difference)) max_rel=$(maximum(relative)) argmax=$(argmax(relative))")
        @test all(isapprox.(actual, expected; rtol=1e-12, atol=1e-12))
    end
end

