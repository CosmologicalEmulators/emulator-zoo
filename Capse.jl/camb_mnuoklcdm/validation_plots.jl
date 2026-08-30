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
    y_max = maximum(filter(isfinite, metrics["dense_knox_p99"]))
    p = plot(
        ell,
        metrics["dense_knox_p99"],
        fillrange=0,
        fillalpha=0.55,
        color=:steelblue,
        linealpha=0,
        label="99%",
        xscale=:log10,
        ylims=(0, 1.05 * y_max),
        xlabel="ℓ",
        ylabel="absolute error / cosmic-variance error",
        title="$spectrum validation: Knox-normalized error",
        grid=true,
    )
    plot!(p, ell, metrics["dense_knox_p95"], fillrange=0,
        fillalpha=0.65, color=:mediumorchid, linealpha=0, label="95%")
    plot!(p, ell, metrics["dense_knox_p68"], fillrange=0,
        fillalpha=0.85, color=:gold, linealpha=0, label="68%")
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
