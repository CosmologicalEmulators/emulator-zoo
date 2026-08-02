module CapseCambMnuW0WaGeneration

using EmulatorsTrainer
using PythonCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DESIGN_SEED
export create_design, initialize_backend, compute_observables, static_axes

const PARAMETER_NAMES = ["ln10As", "ns", "tau", "H0", "omega_b", "omega_c", "Mnu", "w0", "wa"]
const LOWER_BOUNDS = [2.5, 0.85, 0.02, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [3.5, 1.05, 0.15, 90.0, 0.025, 0.16, 0.5, 1.0, 2.0]
const DESIGN_SEED = 20260735
const CAMB_DIRECTORY = @__DIR__

function create_design(n_samples; seed=DESIGN_SEED)
    Random.seed!(seed)
    design = EmulatorsTrainer.create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
    w0 = view(design, 8, :)
    wa = copy(view(design, 9, :))
    for (i, j) in zip(sortperm(w0), sortperm(wa; rev=true))
        design[9, i] = wa[j]
    end
    all(design[8, :] .+ design[9, :] .< 0) || error("w0+wa constraint failed")
    return design
end

function initialize_backend()
    sys = pyimport("sys")
    sys.path.insert(0, CAMB_DIRECTORY)
    return pyimport("camb_worker")
end

function compute_observables(parameters, backend)
    result = backend.compute_spectra(parameters, 9000)
    values = Tuple(
        pyconvert(Vector{Float64}, result[name])
        for name in ("TT", "TE", "EE", "PP", "TT_dense", "TE_dense", "EE_dense", "PP_dense")
    )
    arrays = (
        TT=values[1], TE=values[2], EE=values[3], PP=values[4],
        TT_dense=values[5], TE_dense=values[6], EE_dense=values[7], PP_dense=values[8],
    )
    all(x -> all(isfinite, x), arrays) || error("CAMB output contains NaN or Inf")
    any(arrays.TT_dense .< 0) && error("TT contains negative values")
    any(arrays.EE_dense .< 0) && error("EE contains negative values")
    any(arrays.PP_dense .<= 0) && error("PP contains non-positive values")
    return arrays
end

function static_axes(backend)
    return (
        ell_dense=collect(2.0:9000.0),
        ell_256=pyconvert(Vector{Float64}, backend.lobatto_nodes(256, 2.0, 9000.0)),
        ell_192=pyconvert(Vector{Float64}, backend.lobatto_nodes(192, 2.0, 9000.0)),
    )
end

end
