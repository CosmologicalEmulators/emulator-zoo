using AbstractCosmologicalEmulators
using ArgParse
using EmulatorsTrainer
using JSON3
using NPZ

const PARAMETER_NAMES = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
const OUTPUT_NAMES = ["sigma8", "sigma8_z", "r_drag", "H_z", "r_z", "D_z", "f_z"]

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--dataset"; required=true
        "--artifact"; required=true
        "--basis"; default="ln10As"
        "--output"; required=true
    end
    return parse_args(settings)
end

function main()
    arguments = parse_commandline()
    basis = arguments["basis"]
    basis in ("ln10As", "sigma8") || error("--basis must be ln10As or sigma8")

    dataset = load_hdf5_dataset(abspath(arguments["dataset"]))
    all(dataset.valid) || error("Dataset contains invalid samples")
    dataset.parameter_names == PARAMETER_NAMES || error(
        "Unexpected parameter order: $(dataset.parameter_names)",
    )

    observable_name = Symbol("result_$(basis)_basis")
    ground_truth = dataset.observables[observable_name]
    size(ground_truth, 2) == length(OUTPUT_NAMES) || error("Unexpected output shape")

    emulator = AbstractCosmologicalEmulators.load_trained_emulator(
        abspath(arguments["artifact"]);
        backend=LuxEmulator,
    )
    n_samples = size(dataset.parameters, 1)
    residuals = Matrix{Float64}(undef, n_samples, length(OUTPUT_NAMES))

    for sample_index in 1:n_samples
        parameters = vec(dataset.parameters[sample_index, :])
        input = if basis == "ln10As"
            parameters
        else
            sigma8 = dataset.observables[:result_ln10As_basis][sample_index, 1]
            [parameters[1], sigma8, parameters[3:end]...]
        end
        prediction = Base.invokelatest(
            AbstractCosmologicalEmulators.run_emulator,
            input,
            emulator,
        )
        truth = vec(ground_truth[sample_index, :])
        all(isfinite, prediction) || error("Non-finite prediction at sample $sample_index")
        all(isfinite, truth) || error("Non-finite target at sample $sample_index")
        any(iszero, truth) && error("Zero target at sample $sample_index")
        residuals[sample_index, :] = 100.0 .* abs.(1.0 .- prediction ./ truth)
    end

    percentiles = [64.0, 95.0, 99.0]
    percentile_residuals = EmulatorsTrainer.sort_residuals(
        residuals; percentiles,
    )
    output_directory = abspath(arguments["output"])
    mkpath(output_directory)
    npzwrite(joinpath(output_directory, "residuals_percentiles.npy"), percentile_residuals)
    open(joinpath(output_directory, "residuals_metadata.json"), "w") do stream
        JSON3.write(stream, Dict(
            "dataset" => abspath(arguments["dataset"]),
            "artifact" => abspath(arguments["artifact"]),
            "basis" => basis,
            "n_samples" => n_samples,
            "percentiles" => percentiles,
            "output_names" => OUTPUT_NAMES,
            "residual_definition" => "100 * abs(1 - prediction / ground_truth)",
        ))
    end
    println("Validated $n_samples independent samples for basis=$basis")
    println("Wrote $(joinpath(output_directory, "residuals_percentiles.npy"))")
end

main()
