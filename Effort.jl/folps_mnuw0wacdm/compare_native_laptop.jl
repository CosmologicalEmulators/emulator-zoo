using AbstractCosmologicalEmulators
using Effort
using EmulatorsTrainer
using JSON3
using LinearAlgebra
using NPZ
using PyCall
using Statistics
using Test

const ROOT = @__DIR__
const DATASET_PATH = get(ENV, "FOLPS_DATASET", joinpath(ROOT, "data", "cluster_10k", "dataset.h5"))
const ARTIFACT_ROOT = get(ENV, "FOLPS_ARTIFACT_ROOT", joinpath(ROOT, "artifacts", "cluster_10k"))
const SAMPLE_COUNT = parse(Int, get(ENV, "FOLPS_NATIVE_SAMPLES", "100"))
const Q_PAR = parse(Float64, get(ENV, "FOLPS_Q_PAR", "1.03"))
const Q_PERP = parse(Float64, get(ENV, "FOLPS_Q_PERP", "0.97"))

function growth(parameters)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=parameters["ln10As"], nₛ=parameters["ns"], h=parameters["H0"] / 100,
        ωb=parameters["ombh2"], ωc=parameters["omch2"], mν=parameters["Mnu"],
        w0=parameters["w0"], wa=parameters["wa"])
    return Effort.D_f_z(parameters["z"], cosmology)
end

function nuisance(f0)
    b1 = 1.645
    return [b1, -0.46, -4 / 7 * (b1 - 1), 32 / 315 * (b1 - 1),
        3.0, -28.9, 2.0, 0.2, 0.08, -8.1, 4719.7358, f0]
end

relative_l2(prediction, reference) = norm(prediction - reference) / norm(reference)

function summarize(values)
    return Dict(
        "median" => median(values),
        "p68" => quantile(values, 0.68),
        "p95" => quantile(values, 0.95),
        "p99" => quantile(values, 0.99),
        "maximum" => maximum(values),
    )
end

function main()
    sys = pyimport("sys")
    sys_path = pycall(pybuiltin("getattr"), PyObject, sys, "path")
    pycall(pybuiltin("getattr")(sys_path, "insert"), PyAny, 0, ROOT)
    worker = pyimport("folps_worker")
    basis_module = pyimport("folps_basis")
    folps = pyimport("folps")
    backend = worker.Backend()
    calculator = folps.RSDMultipolesPowerSpectrumCalculator(model="EFT")

    py"""
    import numpy as _np

    def _native_references(backend, calculator, basis_module, parameters, nuisance,
                           qpar, qperp, k_output):
        result, table, table_now = backend.compute(parameters, return_native=True)
        k = _np.asarray(result["k"])
        basis = {ell: result[f"pk_{ell}"] for ell in (0, 2, 4)}
        runtime_contracted = _np.asarray(basis_module.contract_basis(basis, k, nuisance))
        native_nuisance = _np.asarray(nuisance).copy()
        native_nuisance[-1] = float(table[-1])
        native_contracted = _np.asarray(
            basis_module.contract_basis(basis, k, native_nuisance))
        native_parameters = _np.concatenate((_np.asarray(nuisance[:11]), [0.0]))
        intrinsic = _np.asarray(calculator.get_rsd_pkell(
            k, 1.0, 1.0, native_parameters, table, table_now, damping=None))
        distorted = _np.asarray(calculator.get_rsd_pkell(
            _np.asarray(k_output), qpar, qperp, native_parameters,
            table, table_now, damping=None))
        return intrinsic, native_contracted, runtime_contracted, distorted
    """
    native_references = py"_native_references"

    dataset = load_hdf5_dataset(DATASET_PATH)
    k = vec(dataset.axes[:k])
    k_output = k[(k .>= 1.5e-3) .& (k .<= 0.45)]
    validation_indices = Int.(npzread(joinpath(
        ARTIFACT_ROOT, "0", "11", "validation_indices.npy"))) .+ 1
    n_samples = min(SAMPLE_COUNT, length(validation_indices))
    selected_indices = validation_indices[1:n_samples]
    emulators = Dict(ell => Effort.load_multipole_emulator(
        joinpath(ARTIFACT_ROOT, string(ell)) * "/"; emu=SimpleChainsEmulator)
        for ell in (0, 2, 4))

    intrinsic_errors = Dict(ell => Float64[] for ell in (0, 2, 4))
    exact_basis_errors = Dict(ell => Float64[] for ell in (0, 2, 4))
    runtime_f0_errors = Dict(ell => Float64[] for ell in (0, 2, 4))
    external_ap_errors = Dict(ell => Float64[] for ell in (0, 2, 4))
    ap_method_errors = Dict(ell => Float64[] for ell in (0, 2, 4))

    @testset "Laptop emulator versus fresh native Folps" begin
        for (counter, sample_index) in enumerate(selected_indices)
            input = vec(dataset.parameters[sample_index, :])
            parameters = Dict(dataset.parameter_names[j] => input[j] for j in eachindex(input))
            D, f0 = growth(parameters)
            biases = nuisance(f0)
            native_result = pycall(native_references, PyAny, backend, calculator,
                basis_module, parameters, biases, Q_PAR, Q_PERP, k_output)
            native_intrinsic = Matrix{Float64}(native_result[1])
            native_basis_contracted = Matrix{Float64}(native_result[2])
            runtime_basis_contracted = Matrix{Float64}(native_result[3])
            native_internal_ap = Matrix{Float64}(native_result[4])
            emulator_intrinsic = Matrix{Float64}(undef, 3, length(k))

            for (row, ell) in enumerate((0, 2, 4))
                emulator_intrinsic[row, :] = Base.invokelatest(
                    Effort.get_Pℓ, input, D, biases, emulators[ell])
                push!(intrinsic_errors[ell], relative_l2(
                    emulator_intrinsic[row, :], native_intrinsic[row, :]))
                push!(exact_basis_errors[ell], relative_l2(
                    native_basis_contracted[row, :], native_intrinsic[row, :]))
                push!(runtime_f0_errors[ell], relative_l2(
                    runtime_basis_contracted[row, :], native_intrinsic[row, :]))
                @test all(isfinite, emulator_intrinsic[row, :])
                @test exact_basis_errors[ell][end] < 5e-12
            end

            emulator_external_ap = Effort.apply_AP(k, k_output,
                emulator_intrinsic[1, :], emulator_intrinsic[2, :], emulator_intrinsic[3, :],
                Q_PAR, Q_PERP)
            native_external_ap = Effort.apply_AP(k, k_output,
                native_intrinsic[1, :], native_intrinsic[2, :], native_intrinsic[3, :],
                Q_PAR, Q_PERP)
            for (row, ell) in enumerate((0, 2, 4))
                push!(external_ap_errors[ell], relative_l2(
                    emulator_external_ap[row], native_external_ap[row]))
                push!(ap_method_errors[ell], relative_l2(
                    native_external_ap[row], native_internal_ap[row, :]))
            end
            counter % 10 == 0 && println("completed $counter / $n_samples native comparisons")
        end
    end

    summary = Dict(
        "sample_count" => n_samples,
        "sample_indices" => selected_indices,
        "q_par" => Q_PAR,
        "q_perp" => Q_PERP,
        "intrinsic_emulator_vs_native" => Dict(string(ell) => summarize(intrinsic_errors[ell]) for ell in (0, 2, 4)),
        "exact_basis_contraction_vs_native" => Dict(string(ell) => summarize(exact_basis_errors[ell]) for ell in (0, 2, 4)),
        "effort_f0_basis_vs_native_class_f0" => Dict(string(ell) => summarize(runtime_f0_errors[ell]) for ell in (0, 2, 4)),
        "external_ap_emulator_vs_native" => Dict(string(ell) => summarize(external_ap_errors[ell]) for ell in (0, 2, 4)),
        "external_ap_vs_native_internal_ap" => Dict(string(ell) => summarize(ap_method_errors[ell]) for ell in (0, 2, 4)),
    )
    for ell in (0, 2, 4)
        println("ell=$ell intrinsic emulator/native: ", summary["intrinsic_emulator_vs_native"][string(ell)])
        println("ell=$ell external AP emulator/native: ", summary["external_ap_emulator_vs_native"][string(ell)])
        println("ell=$ell external/internal AP methods: ", summary["external_ap_vs_native_internal_ap"][string(ell)])
    end
    open(joinpath(ARTIFACT_ROOT, "laptop_native_comparison.json"), "w") do stream
        JSON3.write(stream, summary)
    end
end

main()
