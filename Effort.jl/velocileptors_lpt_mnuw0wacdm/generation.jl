module VelocileptorsLPTMnuW0WaGeneration

using EmulatorsTrainer
using JSON3
using NPZ
using PyCall
using Random
using SHA

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS
export create_design, initialize_backend, compute_observables, write_sample

const PARAMETER_NAMES = [
    "z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa",
]
const LOWER_BOUNDS = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, 0.5, 2.0]

struct VelocileptorsBackend
    classy::PyObject
    lpt_rsd::PyObject
    konh::Vector{Float64}
    kv::Vector{Float64}
end

function create_design(n_samples::Integer; seed::Integer=20260738)
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
    classy = pyimport("classy")
    lpt_rsd = pyimport("velocileptors.LPT.lpt_rsd_fftw")
    konh = 10.0 .^ range(log10(1.0e-3), log10(10.0); length=20_000)
    kv = vcat(
        5.0e-4,
        10.0 .^ range(log10(1.5e-3), log10(2.5e-2); length=10),
        collect(3.0e-2:1.0e-2:5.0e-1),
    )
    return VelocileptorsBackend(classy, lpt_rsd, konh, kv)
end

function class_parameters(parameters)
    mnu = parameters["Mν"]
    return Dict(
        "output" => "mPk",
        "P_k_max_h/Mpc" => 20.0,
        "z_pk" => "0.0,3.0",
        "h" => parameters["H0"] / 100,
        "omega_b" => parameters["ombh2"],
        "omega_cdm" => parameters["omch2"],
        "ln10^{10}A_s" => parameters["ln10As"],
        "n_s" => parameters["ns"],
        "tau_reio" => 0.0568,
        "N_ur" => mnu > 0 ? 2.033 : 3.046,
        "N_ncdm" => mnu > 0 ? 1 : 0,
        "m_ncdm" => mnu,
        "use_ppf" => "yes",
        "w0_fld" => parameters["w0"],
        "wa_fld" => parameters["wa"],
        "fluid_equation_of_state" => "CLP",
        "cs2_fld" => 1.0,
        "Omega_Lambda" => 0.0,
        "Omega_scf" => 0.0,
    )
end

function compute_observables(parameters, backend::VelocileptorsBackend)
    parameters["w0"] + parameters["wa"] < 0 || throw(ArgumentError("w0 + wa must be negative"))
    h = parameters["H0"] / 100
    z = parameters["z"]
    cosmology = backend.classy.Class()
    try
        cosmology.set(class_parameters(parameters))
        cosmology.compute()
        growth_rate = Float64(cosmology.scale_independent_growth_factor_f(z))
        plin = [Float64(cosmology.pk_cb(k * h, z)) * h^3 for k in backend.konh]
        all(isfinite, plin) || error("CLASS linear power spectrum contains NaN or Inf")

        model = backend.lpt_rsd.LPT_RSD(
            backend.konh,
            plin;
            kIR=0.2,
            use_Pzel=false,
            cutoff=10,
            extrap_min=-4,
            extrap_max=3,
            N=2_000,
            threads=1,
            jn=5,
        )
        model.make_pltable(growth_rate; kv=backend.kv, apar=1.0, aperp=1.0, ngauss=3)
        p0 = Array(model.p0ktable)
        p2 = Array(model.p2ktable)
        p4 = Array(model.p4ktable)
        for (name, values) in (("pk_0", p0), ("pk_2", p2), ("pk_4", p4))
            all(isfinite, values) || error("$name contains NaN or Inf")
        end
        return (kv=Vector{Float64}(model.kv), pk_lin=plin, pk_0=p0, pk_2=p2, pk_4=p4)
    finally
        try
            cosmology.struct_cleanup()
            cosmology.empty()
        catch
        end
    end
end

function sample_id(parameters)
    representation = join(
        (string(name, "=", parameters[name]) for name in PARAMETER_NAMES),
        ";",
    )
    return "sample_" * bytes2hex(sha1(representation))[1:16]
end

function write_sample(root_directory, parameters, observables)
    directory = joinpath(root_directory, sample_id(parameters))
    mkdir(directory)
    npzwrite(joinpath(directory, "kv.npy"), observables.kv)
    npzwrite(joinpath(directory, "pk_lin.npy"), observables.pk_lin)
    npzwrite(joinpath(directory, "pk_0.npy"), observables.pk_0)
    npzwrite(joinpath(directory, "pk_2.npy"), observables.pk_2)
    npzwrite(joinpath(directory, "pk_4.npy"), observables.pk_4)
    record = Dict{String,Any}(parameters)
    record["sample_id"] = basename(directory)
    open(joinpath(directory, "effort_dict.json"), "w") do stream
        JSON3.write(stream, record)
    end
    return directory
end

end
