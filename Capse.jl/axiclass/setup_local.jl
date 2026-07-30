using Pkg
Pkg.activate(@__DIR__)
repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.develop([
    PackageSpec(path=joinpath(repository_root, package))
    for package in ("EmulatorsTrainer.jl", "AbstractCosmologicalEmulators.jl", "Capse.jl")
])
Pkg.instantiate()
