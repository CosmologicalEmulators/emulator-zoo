using Test

include(joinpath(@__DIR__, "generation.jl"))
using .CapseCambMnuOkGeneration

@testset "Mnu-OmegaK-LambdaCDM design" begin
    n_samples = 10_000
    design = create_design(n_samples; seed=20260826)
    repeated = create_design(n_samples; seed=20260826)

    @test design == repeated
    @test size(design) == (length(PARAMETER_NAMES), n_samples)
    @test PARAMETER_NAMES[end] == "OmegaK"
    @test LOWER_BOUNDS[end] == -0.2
    @test UPPER_BOUNDS[end] == 0.2
    @test OUTPUT_LMAX == 9500
    @test all(LOWER_BOUNDS .< minimum(design; dims=2)[:, 1])
    @test all(maximum(design; dims=2)[:, 1] .< UPPER_BOUNDS)

    omega_k = view(design, 8, :)
    omega_k_edges = range(LOWER_BOUNDS[8], UPPER_BOUNDS[8]; length=9)
    for interval in zip(omega_k_edges[1:(end - 1)], omega_k_edges[2:end])
        lower, upper = interval
        in_slice = (lower .<= omega_k) .& (omega_k .< upper)
        @test count(in_slice) > 1000
    end
end
