using Test

bias_combination = include(joinpath(@__DIR__, "biascombination.jl"))
jacobian_bias_combination = include(joinpath(@__DIR__, "jacbiascombination.jl"))
biases = [1.6, -0.4, -0.3, 0.07, 3.0, -20.0, 1.0, 0.2, 0.08, -8.0, 4700.0, 0.82]

@testset "Folps EFT bias functions" begin
    coefficients = bias_combination(biases)
    jacobian = jacobian_bias_combination(biases)
    @test length(coefficients) == 23
    @test size(jacobian) == (23, 12)
    @test all(isfinite, coefficients)
    @test all(isfinite, jacobian)
    for index in eachindex(biases)
        step = 1.0e-6 * max(1.0, abs(biases[index]))
        plus, minus = copy(biases), copy(biases)
        plus[index] += step
        minus[index] -= step
        finite_difference = (bias_combination(plus) - bias_combination(minus)) / (2step)
        @test finite_difference ≈ jacobian[:, index] rtol=2e-9 atol=2e-9
    end
end
