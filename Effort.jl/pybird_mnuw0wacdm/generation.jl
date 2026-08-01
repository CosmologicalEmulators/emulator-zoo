module PyBirdMnuW0WaGeneration

using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, K_GRID, KD_GRID
export create_design, initialize_backend, worker_backend, compute_observables

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
const LOWER_BOUNDS = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, 0.5, 2.0]
const K_GRID = 10.0 .^ range(-5, 0; length=200)
const KD_GRID = collect(0.005:0.004:0.297)

struct Backend
    classy::PyObject
    correlator::PyObject
end

const WORKER_BACKEND = Ref{Union{Nothing,Backend}}(nothing)

PyCall.py"""
import numpy as np
def _pybird_compute(correlator, kk, pk_lin, f):
    correlator.compute({
        "kk": np.asarray(kk),
        "pk_lin": np.asarray(pk_lin),
        "f": f,
    })
    return correlator.bird.P11l, correlator.bird.Ploopl, correlator.bird.Pctl

def _pybird_collect():
    import gc
    gc.collect()
"""

function create_design(n_samples::Integer; seed::Integer=20260763)
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

initialize_backend() = Backend(pyimport("classy"), pyimport("pybird.correlator"))

function worker_backend()
    backend = WORKER_BACKEND[]
    if backend === nothing
        backend = initialize_backend()
        WORKER_BACKEND[] = backend
    end
    return backend
end

function compute_observables(parameters, backend::Backend)
    z = parameters["z"]
    cosmology = backend.classy.Class()
    try
        h = parameters["H0"] / 100
        cosmology.set(Dict(
            "output" => "mPk", "P_k_max_h/Mpc" => 20.0, "z_pk" => "0.0,3.0",
            "h" => h, "omega_b" => parameters["ombh2"], "omega_cdm" => parameters["omch2"],
            "ln10^{10}A_s" => parameters["ln10As"], "n_s" => parameters["ns"],
            "tau_reio" => 0.0568, "N_ur" => 2.033, "N_ncdm" => 1,
            "m_ncdm" => parameters["Mν"], "use_ppf" => "yes",
            "w0_fld" => parameters["w0"], "wa_fld" => parameters["wa"],
            "fluid_equation_of_state" => "CLP", "cs2_fld" => 1.0,
            "Omega_Lambda" => 0.0, "Omega_scf" => 0.0,
        ))
        cosmology.compute()
        pk_lin = [Float64(cosmology.pk_cb(k * h, z)) * h^3 for k in K_GRID]
        f = Float64(cosmology.scale_independent_growth_factor_f(z))
        correlator = backend.correlator.Correlator()
        correlator.set(Dict(
            "output" => "bPk", "multipole" => 3, "kmax" => 0.3,
            "xdata" => KD_GRID, "km" => 0.7, "kr" => 0.35,
            "nd" => 3e-4, "eft_basis" => "eftoflss", "with_stoch" => true,
            "with_bias" => false, "with_resum" => true,
        ))
        pybird_result = py"_pybird_compute"(correlator, K_GRID, pk_lin, f)
        result = (
            P11l=Array(pybird_result[1]),
            Ploopl=Array(pybird_result[2]),
            Pctl=Array(pybird_result[3]),
        )
        all(values -> all(isfinite, values), result) || error("PyBird output contains NaN or Inf")
        py"_pybird_collect"()
        GC.gc()
        return result
    finally
        try
            cosmology.struct_cleanup()
            cosmology.empty()
        catch
        end
    end
end

compute_observables(parameters) = compute_observables(parameters, worker_backend())

end
