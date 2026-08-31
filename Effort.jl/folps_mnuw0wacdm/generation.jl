module FolpsMnuW0WaGeneration

using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DESIGN_SEED
export create_design, initialize_backend, compute_observables, static_axes

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
const LOWER_BOUNDS = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, 0.5, 2.0]
const DESIGN_SEED = 20260831

function create_design(n_samples::Integer; seed::Integer=DESIGN_SEED)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    Random.seed!(seed)
    design = create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
    w0 = view(design, 8, :)
    wa = copy(view(design, 9, :))
    for (w0_index, wa_index) in zip(sortperm(w0), sortperm(wa; rev=true))
        design[9, w0_index] = wa[wa_index]
    end
    all(design[8, :] .+ design[9, :] .< 0) || error("Failed to enforce w0 + wa < 0")
    return design
end

function initialize_backend()
    sys = pyimport("sys")
    builtins = pyimport("builtins")
    python_path = pycall(builtins.getattr, PyObject, sys, "path")
    insert = pycall(builtins.getattr, PyObject, python_path, "insert")
    pycall(insert, PyAny, 0, @__DIR__)
    worker = pyimport("folps_worker")
    return worker.Backend()
end

function compute_observables(parameters, backend)
    result = backend.compute(parameters)
    observables = (
        pk_0=convert(Matrix{Float64}, result["pk_0"]),
        pk_2=convert(Matrix{Float64}, result["pk_2"]),
        pk_4=convert(Matrix{Float64}, result["pk_4"]),
    )
    all(values -> all(isfinite, values), observables) ||
        error("Folps output contains NaN or Inf")
    return observables
end

function static_axes(backend)
    fiducial = Dict(
        "z" => 0.8, "ln10As" => 3.044, "ns" => 0.965, "H0" => 67.36,
        "ombh2" => 0.02237, "omch2" => 0.12, "Mnu" => 0.06,
        "w0" => -1.0, "wa" => 0.0,
    )
    result = backend.compute(fiducial)
    return (k=convert(Vector{Float64}, result["k"]),)
end

end
