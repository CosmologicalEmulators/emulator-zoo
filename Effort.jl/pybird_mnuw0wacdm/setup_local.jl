using Pkg

Pkg.activate(@__DIR__)
try
    Pkg.free("Effort")
catch
end
Pkg.develop(PackageSpec(
    path=normpath(joinpath(@__DIR__, "..", "..", "..", "EmulatorsTrainer.jl")),
))
Pkg.resolve()
Pkg.instantiate()
