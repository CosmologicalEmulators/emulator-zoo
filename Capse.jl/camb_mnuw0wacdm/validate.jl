using Capse
using HDF5
using JSON3
using NPZ
using PyCall
using Statistics

const PARAMETER_NAMES = [
    "ln10As", "ns", "tau", "H0", "omega_b", "omega_c", "Mnu", "w0", "wa",
]

function sample_indices(metadata, n_samples)
    if get(ENV, "CAPSE_VALIDATE_ALL", "0") == "1"
        return collect(1:n_samples)
    end
    haskey(metadata, "validation_sample_ids") ||
        error("Artifact has no validation_sample_ids; set CAPSE_VALIDATE_ALL=1 for an external benchmark")
    return [parse(Int, split(String(id), "_")[2]) for id in metadata["validation_sample_ids"]]
end

function parameter_matrix(parameters, parameter_names)
    locations = Dict(name => findfirst(==(name), parameter_names) for name in PARAMETER_NAMES)
    all(!isnothing, values(locations)) || error("Dataset parameter names do not match train.jl")
    result = Matrix{Float64}(undef, length(PARAMETER_NAMES), size(parameters, 1))
    for (row, name) in enumerate(PARAMETER_NAMES)
        result[row, :] .= parameters[:, locations[name]]
    end
    return result
end

function read_rows(dataset, indices)
    isempty(indices) && error("Validation set is empty")
    first_index, last_index = extrema(indices)
    block = Array(dataset[first_index:last_index, :])
    return block[indices .- first_index .+ 1, :]
end

function load_dataset(path, spectrum, indices)
    h5open(path, "r") do file
        n_samples = size(file["parameters"], 1)
        all((1 .<= indices) .& (indices .<= n_samples)) || error("Validation index outside dataset")
        parameter_names = String.(file["parameter_names"][:])
        parameters = read_rows(file["parameters"], indices)
        ell_dense = Float64.(file["axes/ell_dense"][:])
        truth = permutedims(read_rows(file["observables/$(spectrum)_dense"], indices))
        tt = spectrum == "TE" ? permutedims(read_rows(file["observables/TT_dense"], indices)) : nothing
        ee = spectrum == "TE" ? permutedims(read_rows(file["observables/EE_dense"], indices)) : nothing
        return (
            parameters=parameter_matrix(parameters, parameter_names),
            ell=ell_dense,
            truth=truth,
            tt,
            ee,
        )
    end
end

function cubic_values(source_ell, values, target_ell)
    size(values, 1) == length(source_ell) ||
        error("Interpolation shape mismatch: source ell=$(length(source_ell)), values=$(size(values))")
    spline_values = permutedims(values)
    spline = pyimport("scipy.interpolate").CubicSpline(source_ell, spline_values; axis=1)
    return permutedims(convert(Matrix{Float64}, spline(target_ell)))
end

function finite_stats(values)
    n_ell = size(values, 1)
    output = (p50=fill(NaN, n_ell), p95=fill(NaN, n_ell), maximum=fill(NaN, n_ell))
    for index in 1:n_ell
        row = filter(isfinite, vec(values[index, :]))
        isempty(row) && continue
        output.p50[index] = median(row)
        output.p95[index] = quantile(row, 0.95)
        output.maximum[index] = maximum(row)
    end
    return output
end

function sample_rms(values)
    result = fill(NaN, size(values, 2))
    for index in axes(values, 2)
        column = filter(isfinite, vec(values[:, index]))
        isempty(column) || (result[index] = sqrt(mean(column .^ 2)))
    end
    return result
end

function main()
    length(ARGS) == 3 || error("Usage: validate.jl SPECTRUM DATASET ARTIFACT_DIRECTORY")
    requested_spectrum = uppercase(ARGS[1])
    spectrum = endswith(requested_spectrum, "_LOG") ? chop(requested_spectrum; tail=4) : requested_spectrum
    spectrum in ("TT", "TE", "EE", "PP") || error("Unknown spectrum: $spectrum")
    dataset = abspath(ARGS[2])
    artifact = abspath(ARGS[3])
    metadata = JSON3.read(read(joinpath(artifact, "training_metadata.json"), String))

    n_samples = h5open(dataset, "r") do file
        size(file["parameters"], 1)
    end
    indices = sample_indices(metadata, n_samples)
    data = load_dataset(dataset, spectrum, indices)
    emulator = Capse.load_emulator(artifact)
    prediction = Base.invokelatest(Capse.get_Cℓ, data.parameters, emulator)
    training_ell = Float64.(npzread(joinpath(artifact, "l.npy")))
    prediction_dense, prediction_nodes = if size(prediction, 1) == length(training_ell)
        (cubic_values(training_ell, prediction, data.ell), prediction)
    elseif size(prediction, 1) == length(data.ell)
        (prediction, cubic_values(data.ell, prediction, training_ell))
    else
        error("Emulator output shape $(size(prediction)) matches neither l.npy length " *
              "$(length(training_ell)) nor dense ell length $(length(data.ell))")
    end
    truth_nodes = cubic_values(data.ell, data.truth, training_ell)

    sigma_dense = if spectrum == "TE"
        sqrt.((data.truth .^ 2 .+ data.tt .* data.ee) ./ reshape(2 .* data.ell .+ 1, :, 1))
    else
        sqrt.(2 ./ reshape(2 .* data.ell .+ 1, :, 1)) .* abs.(data.truth)
    end
    sigma_nodes = cubic_values(data.ell, sigma_dense, training_ell)
    dense_error = abs.(prediction_dense .- data.truth) ./ sigma_dense
    node_error = abs.(prediction_nodes .- truth_nodes) ./ sigma_nodes
    dense_stats = finite_stats(dense_error)
    node_stats = finite_stats(node_error)
    dense_sample_rms = sample_rms(dense_error)
    node_sample_rms = sample_rms(node_error)

    npzwrite(joinpath(artifact, "validation_metrics.npz"), Dict(
        "ell_dense" => data.ell,
        "ell_training" => training_ell,
        "dense_knox_p50" => dense_stats.p50,
        "dense_knox_p95" => dense_stats.p95,
        "dense_knox_max" => dense_stats.maximum,
        "node_knox_p50" => node_stats.p50,
        "node_knox_p95" => node_stats.p95,
        "node_knox_max" => node_stats.maximum,
        "dense_sample_rms" => dense_sample_rms,
        "node_sample_rms" => node_sample_rms,
        "sample_indices" => indices,
        "parameters" => data.parameters,
    ))
    npzwrite(joinpath(artifact, "validation_predictions_dense.npy"), prediction_dense)
    report = Dict{String,Any}(
        "spectrum" => spectrum,
        "n_validation" => length(indices),
        "validation_mode" => get(ENV, "CAPSE_VALIDATE_ALL", "0") == "1" ? "external_all" : "training_holdout",
        "node_count" => length(training_ell),
        "dense_knox_median" => median(filter(isfinite, vec(dense_error))),
        "dense_knox_p95" => quantile(filter(isfinite, vec(dense_error)), 0.95),
        "dense_sample_rms_median" => median(filter(isfinite, dense_sample_rms)),
        "dense_sample_rms_p95" => quantile(filter(isfinite, dense_sample_rms), 0.95),
        "node_knox_median" => median(filter(isfinite, vec(node_error))),
        "node_knox_p95" => quantile(filter(isfinite, vec(node_error)), 0.95),
    )
    open(joinpath(artifact, "validation_report.json"), "w") do stream
        JSON3.write(stream, report)
    end
    JSON3.pretty(report)
    println()
end

main()
