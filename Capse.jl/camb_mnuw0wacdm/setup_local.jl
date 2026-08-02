using Pkg

Pkg.activate(@__DIR__)
repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.develop(PackageSpec(path=joinpath(repository_root, "EmulatorsTrainer.jl")))
Pkg.resolve()
Pkg.instantiate()

using Conda
using PyCall
prefix = dirname(dirname(PyCall.python))
python = PyCall.python
needs_repair = try
    pyimport("camb")
    pyimport("scipy")
    false
catch
    true
end
if needs_repair
    site_packages = strip(readchomp(`$python -c 'import site; print(site.getsitepackages()[0])'`))
    for entry in readdir(site_packages; join=true)
        name = basename(entry)
        any(startswith(name, prefix) for prefix in ("camb", "numpy", "scipy", "sympy", "mpmath")) &&
            rm(entry; recursive=true, force=true)
    end
    Conda.add(
        ["camb", "numpy", "scipy", "sympy", "mpmath"], prefix;
        channel="conda-forge", args=`--force-reinstall`,
    )
end
