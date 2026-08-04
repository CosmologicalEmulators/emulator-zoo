ENV["GKSwstype"] = "100"

using JSON3
using NPZ
using Plots

function main()
    length(ARGS) == 1 || error("Usage: validation_plots.jl ARTIFACT_DIRECTORY")
    artifact = abspath(ARGS[1])
    metrics = npzread(joinpath(artifact, "validation_metrics.npz"))
    report = JSON3.read(read(joinpath(artifact, "validation_report.json"), String))
    spectrum = String(report["spectrum"])
    plot_directory = joinpath(artifact, "validation_plots")
    mkpath(plot_directory)

    ell = metrics["ell_dense"]
    training_ell = metrics["ell_training"]
    p = plot(
        ell,
        metrics["dense_knox_p50"],
        label="dense median",
        xscale=:log10,
        yscale=:log10,
        xlabel="ℓ",
        ylabel="absolute error / cosmic-variance error",
        title="$spectrum validation: Knox-normalized error",
        grid=true,
    )
    plot!(p, ell, metrics["dense_knox_p95"], label="dense p95")
    plot!(p, training_ell, metrics["node_knox_p95"], seriestype=:scatter,
        markersize=2, label="node p95")
    hline!(p, [1.0], linestyle=:dash, color=:black, label="1 Knox sigma")
    savefig(p, joinpath(plot_directory, "knox_error_vs_ell.png"))

    rms = filter(isfinite, metrics["dense_sample_rms"])
    p = histogram(
        rms,
        bins=60,
        xlabel="per-sample RMS Knox error",
        ylabel="number of validation samples",
        title="$spectrum validation: sample RMS error",
        label=false,
        grid=true,
    )
    savefig(p, joinpath(plot_directory, "sample_rms_knox_histogram.png"))
end

main()
