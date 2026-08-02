using Pkg

Pkg.activate(@__DIR__)
repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.develop(PackageSpec(path=joinpath(repository_root, "EmulatorsTrainer.jl")))
Pkg.resolve()
Pkg.instantiate()

using PyCall
python = PyCall.python
run(`$python -m pip install camb scipy`)
