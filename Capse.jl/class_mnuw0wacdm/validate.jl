using AbstractCosmologicalEmulators
using ArgParse
using Capse
using EmulatorsTrainer
using JSON3
using NPZ

const PARAMETER_NAMES = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mν", "w0", "wa"]
const SPECTRA = ("TT", "TE", "EE", "PP")

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--dataset"; required=true
        "--artifacts"; required=true
        "--output"; required=true
        "--spectrum"; default="all"
    end
    return parse_args(settings)
end

function main()
    arguments = parse_commandline()
    dataset = EmulatorsTrainer.load_hdf5_dataset(abspath(arguments["dataset"]))
    all(dataset.valid) || error("Validation dataset contains invalid samples")
    dataset.parameter_names == PARAMETER_NAMES || error("Unexpected parameter order")
    n_samples = size(dataset.parameters, 1)
    output_directory = abspath(arguments["output"])
    mkpath(output_directory)
    spectra = arguments["spectrum"] == "all" ? SPECTRA : (arguments["spectrum"],)
    all(spectrum -> spectrum in SPECTRA, spectra) || error("Unknown spectrum")

    for spectrum in spectra
        observable = dataset.observables[Symbol(spectrum)]
        emulator = Capse.load_emulator(
            joinpath(abspath(arguments["artifacts"]), spectrum),
            emu=SimpleChainsEmulator,
        )
        residuals = Matrix{Float64}(undef, n_samples, size(observable, 2))
        for sample_index in 1:n_samples
            input = vec(dataset.parameters[sample_index, :])
            truth = vec(observable[sample_index, :])
            prediction = Base.invokelatest(Capse.get_Cℓ, input, emulator)
            length(prediction) == length(truth) || error("$spectrum prediction length mismatch")
            any(iszero, truth) && error("Zero $spectrum target at sample $sample_index")
            residuals[sample_index, :] = 100.0 .* abs.(1.0 .- prediction ./ truth)
        end
        percentiles = [64.0, 95.0, 99.0]
        result = EmulatorsTrainer.sort_residuals(residuals; percentiles)
        spectrum_output = joinpath(output_directory, spectrum)
        mkpath(spectrum_output)
        npzwrite(joinpath(spectrum_output, "residuals_percentiles.npy"), result)
        open(joinpath(spectrum_output, "residuals_metadata.json"), "w") do stream
            JSON3.write(stream, Dict(
                "dataset" => abspath(arguments["dataset"]),
                "artifact" => joinpath(abspath(arguments["artifacts"]), spectrum),
                "spectrum" => spectrum, "n_samples" => n_samples,
                "percentiles" => percentiles,
                "residual_definition" => "100 * abs(1 - prediction / ground_truth)",
            ))
        end
        println("Validated $spectrum on $n_samples samples")
    end
end

main()
