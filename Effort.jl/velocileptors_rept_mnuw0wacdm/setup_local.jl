using Pkg

Pkg.activate(@__DIR__)
try
    # Remove a stale local Effort development entry when reusing an old
    # environment. Effort remains a normal registry dependency below.
    Pkg.free("Effort")
catch
end
Pkg.develop(PackageSpec(
    path=normpath(joinpath(@__DIR__, "..", "..", "..", "EmulatorsTrainer.jl")),
))
Pkg.resolve()
Pkg.instantiate()
