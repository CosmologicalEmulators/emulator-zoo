module ClassMnuW0WaGeneration
using EmulatorsTrainer, PyCall, Random
export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, create_design
export initialize_backend, compute_observables

const PARAMETER_NAMES = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mν", "w0", "wa"]
const LOWER_BOUNDS = [2.0, 0.8, 50.0, 0.02, 0.08, 0.01, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [3.5, 1.10, 100.0, 0.025, 0.18, 0.15, 0.5, 1.0, 2.0]

struct Backend
    classy::PyObject
end

function create_design(n; seed=20260752)
    Random.seed!(seed)
    design = create_training_dataset(n, LOWER_BOUNDS, UPPER_BOUNDS)
    w0 = view(design, 8, :)
    wa = copy(view(design, 9, :))
    for (i, j) in zip(sortperm(w0), sortperm(wa; rev=true))
        design[9, i] = wa[j]
    end
    all(design[8, :] .+ design[9, :] .< 0) || error("w0+wa constraint failed")
    return design
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
        "m_ncdm" => p["Mν"],
        "use_ppf" => "yes",
        "w0_fld" => p["w0"],
        "wa_fld" => p["wa"],
        "fluid_equation_of_state" => "CLP",
        "cs2_fld" => 1.0,
        "Omega_Lambda" => 0.0,
        "Omega_scf" => 0.0,
        "accurate_lensing" => 1,
        "non_linear" => "hmcode",
    )
end

function compute_observables(p, backend::Backend)
    c = backend.classy.Class()
    try
        c.set(class_parameters(p))
        c.compute()
        cl = c.lensed_cl(10000)
        ell = collect(0.0:(length(cl["tt"]) - 1))
        factor = ell .* (ell .+ 1) ./ (2π)
        tt = 7.42715e12 .* (factor .* Vector{Float64}(cl["tt"]))[3:10000]
        ee = 7.42715e12 .* (factor .* Vector{Float64}(cl["ee"]))[3:10000]
        te = 7.42715e12 .* (factor .* Vector{Float64}(cl["te"]))[3:10000]
        pp = (ell .* (ell .+ 1) .* ell .* (ell .+ 1) .* Vector{Float64}(cl["pp"]) ./ (2π))[3:10000]
        result = (TT=tt, EE=ee, TE=te, PP=pp)
        all(x -> all(isfinite, x), result) || error("CLASS output contains NaN or Inf")
        return result
    finally
        try
            c.struct_cleanup()
            c.empty()
        catch
        end
    end
end

end
