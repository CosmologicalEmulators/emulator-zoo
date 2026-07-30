using Test

include(joinpath(@__DIR__, "..", "generation.jl"))
using .ACEClassMnuW0WaGeneration

input_line = only(filter(
    line -> !startswith(line, "#") && !isempty(strip(line)),
    readlines(joinpath(@__DIR__, "reference_inputs.txt")),
))
values = parse.(Float64, split(strip(input_line)))
parameters = Dict(ACEClassMnuW0WaGeneration.PARAMETER_NAMES .=> values)
reference = Dict{Symbol,Vector{Float64}}()
for line in filter(line -> !startswith(line, "#") && !isempty(strip(line)), readlines(joinpath(@__DIR__, "reference_outputs.txt")))
    fields = split(line)
    reference[Symbol(fields[1])] = parse.(Float64, fields[2:end])
end

result = ACEClassMnuW0WaGeneration.compute_observables(parameters, ACEClassMnuW0WaGeneration.initialize_backend())
@testset "ACE CLASS Mnu-w0-wa reference" begin
    for name in propertynames(result)
        actual = getproperty(result, name)
        expected = reference[name]
        difference = abs.(actual .- expected)
        relative = difference ./ max.(abs.(expected), eps(Float64))
        println("$name: max_abs=$(maximum(difference)) max_rel=$(maximum(relative)) argmax=$(argmax(relative))")
        @test all(isapprox.(actual, expected; rtol=1e-10, atol=1e-10))
    end
end
