using Pkg

Pkg.activate(@__DIR__)
cmbcheb_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
cosmological_emulators_root = normpath(joinpath(cmbcheb_root, ".."))
trainer_path = joinpath(cmbcheb_root, "EmulatorsTrainer.jl")
effort_path = joinpath(cosmological_emulators_root, "Effort.jl")
isdir(trainer_path) && Pkg.develop(PackageSpec(path=trainer_path))
isdir(effort_path) && Pkg.develop(PackageSpec(path=effort_path))
Pkg.resolve()
Pkg.instantiate()
