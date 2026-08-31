using Test

include(joinpath(@__DIR__, "train.jl"))

@testset "EE hybrid low-ell training grid" begin
    nodes = hybrid_lowell_nodes(256; dense_lower=2, dense_upper=20, upper=9500.0)

    @test default_dense_lowell_max("EE") == 20
    @test default_dense_lowell_max("TT") == 0
    @test default_dense_lowell_max("TE") == 0
    @test default_dense_lowell_max("BB") == 0
    @test default_dense_lowell_max("PP") == 0
    @test length(nodes) == 274
    @test nodes[1:19] == collect(2.0:20.0)
    @test nodes[20] > 20.0
    @test nodes[end] == 9500.0
    @test all(diff(nodes) .> 0)
    @test count(<=(20.0), nodes) == 19
end

@testset "Standard Lobatto training grid" begin
    nodes = lobatto_nodes(256; lower=2.0, upper=9500.0)

    @test length(nodes) == 256
    @test nodes[1] == 2.0
    @test nodes[end] == 9500.0
    @test all(diff(nodes) .> 0)
end
