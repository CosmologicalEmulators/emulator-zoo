using Capse
using JSON3
using NPZ
using Statistics

const PARAMETER_NAMES = ["ln10As", "ns", "tau", "H0", "omega_b", "omega_c"]

function sample_map(data_directory)
    return Dict(
        name => joinpath(data_directory, name)
        for name in sort(readdir(data_directory)) if
        startswith(name, "sample_") && isdir(joinpath(data_directory, name))
    )
end

function main()
    spectrum = uppercase(ARGS[1])
    data_directory = abspath(ARGS[2])
    artifact = abspath(ARGS[3])
    metadata = JSON3.read(read(joinpath(artifact, "training_metadata.json"), String))
    directories = sample_map(data_directory)
    ids = String.(metadata["validation_sample_ids"])
    parameters = Matrix{Float64}(undef, length(PARAMETER_NAMES), length(ids))
    truth = Matrix{Float64}(undef, 8999, length(ids))
    for (column, sample_id) in enumerate(ids)
        directory = directories[sample_id]
        record = JSON3.read(read(joinpath(directory, "params.json"), String))
        parameters[:, column] .= [record[name] for name in PARAMETER_NAMES]
        truth[:, column] .= npzread(joinpath(directory, "$(spectrum)_dense.npy"))
    end
    emulator = Capse.load_emulator(artifact; interpolation=:auto)
    prediction = Base.invokelatest(Capse.get_Cℓ, parameters, emulator)
    size(prediction) == size(truth) || error("Prediction and truth shapes differ")
    all(isfinite, prediction) || error("Prediction contains NaN or Inf")
    absolute = abs.(prediction .- truth)
    scale = maximum(abs.(truth); dims=1)
    relative_mask = abs.(truth) .> 1.0e-8 .* scale
    relative = absolute[relative_mask] ./ abs.(truth[relative_mask])
    report = Dict{String,Any}(
        "spectrum" => spectrum,
        "n_validation" => length(ids),
        "mean_absolute" => mean(absolute),
        "median_absolute" => median(vec(absolute)),
        "p95_absolute" => quantile(vec(absolute), 0.95),
        "max_absolute" => maximum(absolute),
        "mean_relative" => mean(relative),
        "median_relative" => median(relative),
        "p95_relative" => quantile(relative, 0.95),
        "max_relative" => maximum(relative),
    )
    npzwrite(joinpath(artifact, "validation_predictions_capse_dense.npy"), prediction)
    open(joinpath(artifact, "validation_report.json"), "w") do stream
        JSON3.write(stream, report)
    end
    JSON3.pretty(report)
    println()
end

main()
