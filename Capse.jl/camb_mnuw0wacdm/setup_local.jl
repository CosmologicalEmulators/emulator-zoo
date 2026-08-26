using Pkg

Pkg.activate(@__DIR__)
repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.develop(PackageSpec(path=joinpath(repository_root, "EmulatorsTrainer.jl")))
Pkg.resolve()
Pkg.instantiate()

using Conda
using PyCall
prefix = dirname(dirname(PyCall.python))
needs_repair = try
    pyimport("numpy")
    pyimport("scipy")
    pyimport("sympy")
    false
catch
    true
end
if needs_repair
    Conda.add(
        ["numpy", "scipy", "sympy", "mpmath", "packaging"], prefix;
        channel="conda-forge",
    )
end

include(joinpath(@__DIR__, "generation.jl"))
using .CapseCambMnuW0WaGeneration

backend = CapseCambMnuW0WaGeneration.initialize_backend()
configuration = CapseCambMnuW0WaGeneration.backend_configuration(backend)
println("PyCall Python: ", PyCall.python)
println("CAMB path: ", configuration["camb_path"])
println("CAMB version: ", configuration["camb_version"])
println("Recombination model: ", configuration["recombination_model"])
