module ACEClassMnuW0WaGeneration

using Effort
using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS
export create_design, initialize_backend, worker_backend, compute_observables

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
const LOWER_BOUNDS = [0.0, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [5.0, 3.7, 1.10, 90.0, 0.025, 0.18, 0.5, 0.5, 2.0]

struct Backend
    classy::PyObject
end

const WORKER_BACKEND = Ref{Union{Nothing,Backend}}(nothing)

function create_design(n_samples::Integer; seed::Integer=20260760)
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

initialize_backend() = Backend(pyimport("classy"))

function worker_backend()
    backend = WORKER_BACKEND[]
    if backend === nothing
        backend = initialize_backend()
        WORKER_BACKEND[] = backend
    end
    return backend
end

function class_parameters(parameters)
    settings = Dict(
        "output" => "mPk",
        "P_k_max_h/Mpc" => 20.0,
        "z_pk" => "0.0,3.0",
        "h" => parameters["H0"] / 100,
        "omega_b" => parameters["ombh2"],
        "omega_cdm" => parameters["omch2"],
        "ln10^{10}A_s" => parameters["ln10As"],
        "n_s" => parameters["ns"],
        "tau_reio" => 0.0568,
        "N_ur" => 2.033,
        "N_ncdm" => 1,
        "m_ncdm" => parameters["Mν"],
        "use_ppf" => "yes",
        "w0_fld" => parameters["w0"],
        "wa_fld" => parameters["wa"],
        "fluid_equation_of_state" => "CLP",
        "cs2_fld" => 1.0,
        "Omega_Lambda" => 0.0,
        "Omega_scf" => 0.0,
    )
    if parameters["z"] > 3
        settings["z_max_pk"] = parameters["z"]
    end
    return settings
end

function compute_observables(parameters, backend::Backend)
    z = parameters["z"]
    h = parameters["H0"] / 100
    omega_cb = parameters["ombh2"] + parameters["omch2"]
    Omega_cb = omega_cb / h^2
    cosmology = backend.classy.Class()
    try
        cosmology.set(class_parameters(parameters))
        cosmology.compute()
        sigma8 = Float64(cosmology.sigma8)
        sigma8_z = Float64(cosmology.sigma(8.0 / h, z))
        r_drag = Float64(cosmology.rs_drag)
        H_z = Float64(cosmology.Hubble(z)) * 299792.458
        r_z = Float64(cosmology.comoving_distance(z))
        D_z, f_z = Effort.D_f_z(
            z, Omega_cb, h;
            mν=parameters["Mν"], w0=parameters["w0"], wa=parameters["wa"],
        )
        result = (
            result_sigma8_basis=[parameters["ln10As"], sigma8_z, r_drag, H_z, r_z, D_z, f_z],
            result_ln10As_basis=[sigma8, sigma8_z, r_drag, H_z, r_z, D_z, f_z],
        )
        all(values -> all(isfinite, values), result) || error("ACE output contains NaN or Inf")
        return result
    catch error
        if error isa PyCall.PyError
            throw(ErrorException("CLASS failed: $(sprint(showerror, error))"))
        end
        rethrow()
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
