module VelocileptorsREPTMnuW0WaGeneration

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
    rept::PyObject
    pnw::PyObject
    konh::Vector{Float64}
    kv::Vector{Float64}
end

function create_design(n_samples::Integer; seed::Integer=20260744)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    Random.seed!(seed)
    design = create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
    w0 = view(design, 8, :)
    wa = copy(view(design, 9, :))
    for (w0_index, wa_index) in zip(sortperm(w0), sortperm(wa; rev=true))
        design[9, w0_index] = wa[wa_index]
    end
    all(design[8, :] .+ design[9, :] .< 0) ||
        error("Failed to enforce w0 + wa < 0")
    return design
end

function initialize_backend()
    classy = pyimport("classy")
    rept = pyimport("velocileptors.EPT.ept_fullresum_fftw")
    pnw = pyimport("velocileptors.Utils.pnw_dst")
    konh = 10.0 .^ range(-3, 1; length=20_000)
    kv = 10.0 .^ range(log10(5.0e-4), log10(0.5); length=80)
    return VelocileptorsBackend(classy, rept, pnw, konh, kv)
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
    parameters["w0"] + parameters["wa"] < 0 ||
        throw(ArgumentError("w0 + wa must be negative"))
    cosmology = backend.classy.Class()
    try
        cosmology.set(class_parameters(parameters))
        cosmology.compute()
        z = parameters["z"]
        h = parameters["H0"] / 100
        growth_rate = Float64(cosmology.scale_independent_growth_factor_f(z))
        plin = [Float64(cosmology.pk_cb(k * h, z)) * h^3 for k in backend.konh]
        all(isfinite, plin) || error("CLASS linear power spectrum contains NaN or Inf")

        knw, pnw = backend.pnw.pnw_dst(backend.konh, plin)
        model = backend.rept.REPT(
            knw,
            plin;
            pnw=pnw,
            kmin=5.0e-4,
            kmax=0.5,
            nk=80,
            beyond_gauss=true,
            one_loop=true,
            N=2_000,
            extrap_min=-6,
            extrap_max=2,
            cutoff=100,
            threads=1,
        )
        model.compute_redshift_space_power_multipoles_tables(
            growth_rate;
            apar=1.0,
            aperp=1.0,
            ngauss=4,
        )
        observables = (
            kv=Vector{Float64}(model.kv),
            pk_lin=plin,
            pk_0=Array(model.p0ktable),
            pk_2=Array(model.p2ktable),
            pk_4=Array(model.p4ktable),
            knw=Vector{Float64}(knw),
            Pnw=Vector{Float64}(pnw),
        )
        for (name, values) in pairs(observables)
            all(isfinite, values) || error("$name contains NaN or Inf")
        end
        return observables
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
    for name in (:kv, :pk_lin, :pk_0, :pk_2, :pk_4, :knw, :Pnw)
        npzwrite(joinpath(directory, "$name.npy"), getproperty(observables, name))
    end
    record = Dict{String,Any}(parameters)
    record["sample_id"] = basename(directory)
    open(joinpath(directory, "effort_dict.json"), "w") do stream
        JSON3.write(stream, record)
    end
    return directory
end

end
