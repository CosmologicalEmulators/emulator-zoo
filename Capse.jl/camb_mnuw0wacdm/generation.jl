module CapseCambMnuW0WaGeneration

using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DESIGN_SEED, EARLY_W_MAX, OUTPUT_LMAX
export create_design, initialize_backend, backend_configuration, compute_observables, static_axes

const PARAMETER_NAMES = ["ln10As", "ns", "tau", "H0", "omega_b", "omega_c", "Mnu", "w0", "wa"]
const LOWER_BOUNDS = [2.5, 0.85, 0.02, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [3.5, 1.05, 0.15, 90.0, 0.025, 0.16, 0.5, 0.5, 2.0]
const DESIGN_SEED = 20260735
const EARLY_W_MAX = -0.5
const OUTPUT_LMAX = 9500
const CAMB_DIRECTORY = @__DIR__

function create_design(n_samples; seed=DESIGN_SEED)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    Random.seed!(seed)

    accepted_batches = Matrix{Float64}[]
    n_accepted = 0
    while n_accepted < n_samples
        n_remaining = n_samples - n_accepted
        # The constraint retains about 74.3% of the rectangular w0-wa domain.
        # A conservative 65% estimate normally completes this in one batch.
        n_candidates = max(ceil(Int, n_remaining / 0.65), 32)
        candidates = EmulatorsTrainer.create_training_dataset(
            n_candidates, LOWER_BOUNDS, UPPER_BOUNDS,
        )
        valid = vec(candidates[8, :] .+ candidates[9, :] .< EARLY_W_MAX)
        accepted = candidates[:, valid]
        push!(accepted_batches, accepted)
        n_accepted += size(accepted, 2)
    end

    accepted = reduce(hcat, accepted_batches)
    design = accepted[:, randperm(size(accepted, 2))[1:n_samples]]
    all(design[8, :] .+ design[9, :] .< EARLY_W_MAX) ||
        error("w0+wa constraint failed")
    return design
end

function initialize_backend()
    sys = pyimport("sys")
    builtins = pyimport("builtins")
    python_path = pycall(builtins.getattr, PyObject, sys, "path")
    insert = pycall(builtins.getattr, PyObject, python_path, "insert")
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
