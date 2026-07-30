module AxiclassGeneration
using EmulatorsTrainer, PyCall, Random
export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, create_design
export compute_observables

const PARAMETER_NAMES = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "fede", "scf", "log10axion"]
const LOWER_BOUNDS = [2.5, 0.8, 50.0, 0.02, 0.09, 0.01, 1e-4, 0.0, -4.5]
const UPPER_BOUNDS = [3.5, 1.10, 90.0, 0.025, 0.18, 0.15, 0.5, π, -3.0]

function create_design(n; seed=20260740)
    Random.seed!(seed)
    return create_training_dataset(n, LOWER_BOUNDS, UPPER_BOUNDS)
end

function compute_observables(parameters)
    pya = pyimport("pyclass.axiclass")
    cosmo_params = Dict(
        "output" => "tCl pCl lCl",
        "l_max_scalars" => 3000,
        "lensing" => "yes",
        "h" => parameters["H0"] / 100,
        "omega_b" => parameters["ombh2"],
        "omega_cdm" => parameters["omch2"],
        "ln10^{10}A_s" => parameters["ln10As"],
        "n_s" => parameters["ns"],
        "tau_reio" => parameters["τ"],
        "N_ur" => 2.0308,
        "N_ncdm" => 1,
        "m_ncdm" => 0.06,
        "fraction_axion_ac" => parameters["fede"],
        "scf_parameters" => [parameters["scf"], 0.0],
        "log10_axion_ac" => parameters["log10axion"],
        "do_shooting" => true,
        "do_shooting_scf" => true,
        "scf_potential" => "axion",
        "n_axion" => 3,
        "security_small_Omega_scf" => 0.001,
        "n_axion_security" => 2.09,
        "use_big_theta_scf" => true,
        "scf_has_perturbations" => true,
        "attractor_ic_scf" => false,
        "scf_tuning_index" => 0,
        "include_scf_in_delta_m" => true,
        "include_scf_in_delta_cb" => true,
        "scf_evolve_as_fluid" => false,
        "scf_evolve_like_axionCAMB" => false,
    )
    cosmo = pya.ClassEngine(cosmo_params)
    harmonic = pya.Harmonic(cosmo)
    cl = harmonic.lensed_cl()
    ell = pyconvert(Vector{Float64}, pyimport("numpy").arange(length(cl["tt"])))
    factor = ell .* (ell .+ 1) ./ (2π)
    result = (
        TT=7.42715e12 .* (factor .* pyconvert(Vector{Float64}, cl["tt"]))[3:3001],
        EE=7.42715e12 .* (factor .* pyconvert(Vector{Float64}, cl["ee"]))[3:3001],
        TE=7.42715e12 .* (factor .* pyconvert(Vector{Float64}, cl["te"]))[3:3001],
    )
    all(x -> all(isfinite, x), result) || error("axiclass output contains NaN or Inf")
    return result
end

end
