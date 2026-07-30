module ClassLCDMGeneration
using EmulatorsTrainer, PyCall, Random
export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, create_design
export initialize_backend, compute_observables

const PARAMETER_NAMES = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ"]
const LOWER_BOUNDS = [2.0, 0.8, 50.0, 0.02, 0.09, 0.01]
const UPPER_BOUNDS = [3.5, 1.10, 90.0, 0.025, 0.18, 0.20]

struct Backend
    classy::PyObject
end

function create_design(n; seed=20260750)
    Random.seed!(seed)
    return create_training_dataset(n, LOWER_BOUNDS, UPPER_BOUNDS)
end

initialize_backend() = Backend(pyimport("classy"))

function class_parameters(p)
    return Dict(
        "output" => "tCl pCl lCl",
        "l_max_scalars" => 10000,
        "lensing" => "yes",
        "h" => p["H0"] / 100,
        "omega_b" => p["ombh2"],
        "omega_cdm" => p["omch2"],
        "ln10^{10}A_s" => p["ln10As"],
        "n_s" => p["ns"],
        "tau_reio" => p["τ"],
        "N_ur" => 2.0308,
        "N_ncdm" => 1,
        "m_ncdm" => 0.06,
        "use_ppf" => "yes",
        "w0_fld" => -1.0,
        "wa_fld" => 0.0,
        "fluid_equation_of_state" => "CLP",
        "cs2_fld" => 1.0,
        "Omega_Lambda" => 0.0,
        "Omega_scf" => 0.0,
        "accurate_lensing" => 1,
        "non_linear" => "hmcode",
    )
end

function compute_observables(p, backend::Backend)
    cosmo = backend.classy.Class()
    try
        cosmo.set(class_parameters(p))
        cosmo.compute()
        cl = cosmo.lensed_cl(10000)
        ell = collect(0.0:(length(cl["tt"]) - 1))
        factor = ell .* (ell .+ 1) ./ (2π)
        result = (
            TT=7.42715e12 .* (factor .* Vector{Float64}(cl["tt"]))[3:10000],
            EE=7.42715e12 .* (factor .* Vector{Float64}(cl["ee"]))[3:10000],
            TE=7.42715e12 .* (factor .* Vector{Float64}(cl["te"]))[3:10000],
            PP=(ell .* (ell .+ 1) .* ell .* (ell .+ 1) .* Vector{Float64}(cl["pp"]) ./ (2π))[3:10000],
        )
        all(x -> all(isfinite, x), result) || error("CLASS output contains NaN or Inf")
        return result
    finally
        try
            cosmo.struct_cleanup()
            cosmo.empty()
        catch
        end
    end
end

end
