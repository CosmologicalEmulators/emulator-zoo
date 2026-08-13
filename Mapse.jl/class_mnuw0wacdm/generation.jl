module MapseClassMnuW0WaGeneration

using EmulatorsTrainer
using PyCall
using Random

export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, K_GRID, FIXED_LN10AS, FIXED_NS
export create_design, initialize_backend, worker_backend, compute_observables

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
const LOWER_BOUNDS = [0.0, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [5.0, 90.0, 0.025, 0.18, 0.5, 1.0, 2.0]
const FIXED_LN10AS = 3.044
const FIXED_NS = 0.965
const K_GRID = collect(10.0 .^ range(log10(5.0e-6), log10(100.0); length=300))

struct Backend
    classy::PyObject
end

const WORKER_BACKEND = Ref{Union{Nothing,Backend}}(nothing)

function create_design(n_samples::Integer; seed::Integer=20260763)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    Random.seed!(seed)
    sampled_design = create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
    w0 = view(sampled_design, 6, :)
    wa = copy(view(sampled_design, 7, :))
    for (w0_index, wa_index) in zip(sortperm(w0), sortperm(wa; rev=true))
        sampled_design[7, w0_index] = wa[wa_index]
    end
    all(sampled_design[6, :] .+ sampled_design[7, :] .< 0) ||
        error("Failed to enforce w0 + wa < 0")
    design = Matrix{Float64}(undef, length(PARAMETER_NAMES), n_samples)
    design[1, :] .= sampled_design[1, :]
    design[2, :] .= FIXED_LN10AS
    design[3, :] .= FIXED_NS
    design[4:end, :] .= sampled_design[2:end, :]
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
    h = parameters["H0"] / 100
    return Dict(
        "output" => "mPk",
        "P_k_max_h/Mpc" => maximum(K_GRID) / h,
        "z_pk" => "0.0,5.0",
        "h" => h,
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
end

function compute_observables(parameters, backend::Backend)
    z = parameters["z"]
    cosmology = backend.classy.Class()
    try
        cosmology.set(class_parameters(parameters))
        cosmology.compute()
        result = (
            Pk_lin_mm=[Float64(cosmology.pk_lin(k, z)) for k in K_GRID],
            Pk_lin_cb=[Float64(cosmology.pk_cb_lin(k, z)) for k in K_GRID],
        )
        all(values -> all(isfinite, values), result) || error("CLASS linear spectra contain NaN or Inf")
        all(values -> all(>(0), values), result) || error("CLASS linear spectra must be positive")
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
