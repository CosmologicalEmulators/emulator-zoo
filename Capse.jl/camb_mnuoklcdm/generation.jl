module CapseCambMnuOkGeneration

using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DESIGN_SEED, OUTPUT_LMAX
export create_design, initialize_backend, backend_configuration, compute_observables, static_axes

const PARAMETER_NAMES = ["ln10As", "ns", "tau", "H0", "omega_b", "omega_c", "Mnu", "OmegaK"]
const LOWER_BOUNDS = [2.5, 0.85, 0.02, 50.0, 0.02, 0.08, 0.0, -0.2]
const UPPER_BOUNDS = [3.5, 1.05, 0.15, 90.0, 0.025, 0.16, 0.5, 0.2]
const DESIGN_SEED = 20260735
const OUTPUT_LMAX = 9500
const CAMB_DIRECTORY = @__DIR__
const DEFAULT_CAMB_SOURCE = normpath(joinpath(@__DIR__, "..", "..", "..", "tools", "CAMB-cosmorec"))

function create_design(n_samples; seed=DESIGN_SEED)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    Random.seed!(seed)

    return EmulatorsTrainer.create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
end

function initialize_backend()
    sys = pyimport("sys")
    builtins = pyimport("builtins")
    python_path = pycall(builtins.getattr, PyObject, sys, "path")
    insert = pycall(builtins.getattr, PyObject, python_path, "insert")
    camb_source = get(ENV, "CAPSE_CAMB_SOURCE", DEFAULT_CAMB_SOURCE)
    isdir(camb_source) && pycall(insert, PyAny, 0, camb_source)
    pycall(insert, PyAny, 0, CAMB_DIRECTORY)
    backend = pyimport("camb_worker")
    backend.backend_configuration()
    Int(backend.OUTPUT_LMAX) == OUTPUT_LMAX || error("Julia/Python output lmax mismatch")
    return backend
end

function backend_configuration(backend)
    configuration = backend.backend_configuration()
    return Dict{String,Any}(
        "camb_version" => string(configuration["camb_version"]),
        "camb_path" => string(configuration["camb_path"]),
        "recombination_model" => string(configuration["recombination_model"]),
        "helium_fraction" => string(configuration["helium_fraction"]),
        "lens_margin" => Int(configuration["lens_margin"]),
        "output_lmax" => Int(configuration["output_lmax"]),
    )
end

function compute_observables(parameters, backend)
    result = backend.compute_spectra(parameters, OUTPUT_LMAX)
    values = Tuple(
        convert(Vector{Float64}, result[name])
        for name in ("TT_dense", "TE_dense", "EE_dense", "BB_dense", "PP_dense")
    )
    arrays = (
        TT_dense=values[1], TE_dense=values[2], EE_dense=values[3],
        BB_dense=values[4], PP_dense=values[5],
    )
    all(x -> all(isfinite, x), arrays) || error("CAMB output contains NaN or Inf")
    any(arrays.TT_dense .<= 0) && error("TT contains non-positive values")
    any(arrays.EE_dense .<= 0) && error("EE contains non-positive values")
    any(arrays.BB_dense .<= 0) && error("BB contains non-positive values")
    any(arrays.PP_dense .<= 0) && error("PP contains non-positive values")
    return arrays
end

function static_axes(backend)
    return (
        ell_dense=collect(2.0:OUTPUT_LMAX),
        ell_256=convert(Vector{Float64}, backend.lobatto_nodes(256, 2.0, OUTPUT_LMAX)),
        ell_192=convert(Vector{Float64}, backend.lobatto_nodes(192, 2.0, OUTPUT_LMAX)),
    )
end

end
