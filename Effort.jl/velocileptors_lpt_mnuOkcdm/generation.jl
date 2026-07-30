module VelocileptorsLPTMnuOmegaKGeneration

using EmulatorsTrainer, JSON3, NPZ, PyCall, Random, SHA

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS
export create_design, initialize_backend, compute_observables, write_sample

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "Omega_k"]
const LOWER_BOUNDS = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -0.2]
const UPPER_BOUNDS = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, 0.2]

struct Backend
    classy::PyObject
    lpt_rsd::PyObject
    konh::Vector{Float64}
    kv::Vector{Float64}
end

function create_design(n::Integer; seed::Integer=20260741)
    Random.seed!(seed)
    return create_training_dataset(n, LOWER_BOUNDS, UPPER_BOUNDS)
end

function initialize_backend()
    konh = 10.0 .^ range(-3, 1; length=20_000)
    kv = vcat(5e-4, 10.0 .^ range(log10(1.5e-3), log10(2.5e-2); length=10), collect(0.03:0.01:0.50))
    return Backend(pyimport("classy"), pyimport("velocileptors.LPT.lpt_rsd_fftw"), konh, kv)
end

function class_parameters(p)
    return Dict(
        "output" => "mPk", "P_k_max_h/Mpc" => 20.0, "z_pk" => "0.0,3.0",
        "h" => p["H0"] / 100, "omega_b" => p["ombh2"], "omega_cdm" => p["omch2"],
        "ln10^{10}A_s" => p["ln10As"], "n_s" => p["ns"], "tau_reio" => 0.0568,
        "N_ur" => 2.033, "N_ncdm" => 1, "m_ncdm" => p["Mν"],
        "use_ppf" => "yes", "w0_fld" => -1.0, "wa_fld" => 0.0,
        "fluid_equation_of_state" => "CLP", "cs2_fld" => 1.0,
        "Omega_k" => p["Omega_k"], "Omega_Lambda" => 0.0,
    )
end

function compute_observables(p, backend::Backend)
    h, z = p["H0"] / 100, p["z"]
    cosmo = backend.classy.Class()
    try
        cosmo.set(class_parameters(p)); cosmo.compute()
        f = Float64(cosmo.scale_independent_growth_factor_f(z))
        plin = [Float64(cosmo.pk_cb(k * h, z)) * h^3 for k in backend.konh]
        model = backend.lpt_rsd.LPT_RSD(
            backend.konh, plin; kIR=0.2, use_Pzel=false, cutoff=10,
            extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5,
        )
        model.make_pltable(f; kv=backend.kv, apar=1.0, aperp=1.0, ngauss=3)
        result = (
            kv=Vector{Float64}(model.kv), pk_lin=plin,
            pk_0=Array(model.p0ktable), pk_2=Array(model.p2ktable), pk_4=Array(model.p4ktable),
        )
        all(values -> all(isfinite, values), result) || error("Generated arrays contain NaN or Inf")
        return result
    finally
        try cosmo.struct_cleanup(); cosmo.empty() catch end
    end
end

function sample_id(p)
    text = join(("$name=$(p[name])" for name in PARAMETER_NAMES), ";")
    return "sample_" * bytes2hex(sha1(text))[1:16]
end

function write_sample(root, p, result)
    directory = joinpath(root, sample_id(p)); mkdir(directory)
    for name in (:kv, :pk_lin, :pk_0, :pk_2, :pk_4)
        npzwrite(joinpath(directory, "$(name).npy"), getproperty(result, name))
    end
    record = Dict{String,Any}(p); record["sample_id"] = basename(directory)
    open(joinpath(directory, "effort_dict.json"), "w") do io; JSON3.write(io, record); end
    return directory
end

end
