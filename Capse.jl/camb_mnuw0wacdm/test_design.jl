using Statistics
using Test

include(joinpath(@__DIR__, "generation.jl"))
using .CapseCambMnuW0WaGeneration

@testset "Constrained Mnu-w0-wa design" begin
    n_samples = 10_000
    design = create_design(n_samples; seed=20260826)
    repeated = create_design(n_samples; seed=20260826)

    @test design == repeated
    @test size(design) == (length(PARAMETER_NAMES), n_samples)
    @test UPPER_BOUNDS[8] == 0.5
    @test EARLY_W_MAX == -0.5
    @test all(LOWER_BOUNDS .< minimum(design; dims=2)[:, 1])
    @test all(maximum(design; dims=2)[:, 1] .< UPPER_BOUNDS)

    w0 = view(design, 8, :)
    wa = view(design, 9, :)
    @test all(w0 .+ wa .< EARLY_W_MAX)
    @test abs(cor(w0, wa)) < 0.95
    @test std(wa .+ 1.25 .* w0) > 0.25

    w0_edges = range(LOWER_BOUNDS[8], UPPER_BOUNDS[8]; length=9)
    for interval in zip(w0_edges[1:(end - 1)], w0_edges[2:end])
        lower, upper = interval
        in_slice = (lower .<= w0) .& (w0 .< upper)
        @test count(in_slice) > 100
        @test maximum(wa[in_slice]) - minimum(wa[in_slice]) > 0.5
    end
end
