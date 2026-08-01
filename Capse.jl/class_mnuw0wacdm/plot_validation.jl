using ArgParse
using NPZ
using Plots

settings = ArgParseSettings()
@add_arg_table settings begin
    "--residuals"; required=true
    "--output"; required=true
    "--spectrum"; default="TT"
end
arguments = parse_args(settings)

residuals = npzread(abspath(arguments["residuals"]))
ell = 2:(size(residuals, 2) + 1)
plot(
    ell, residuals[1, :];
    label="64%", linewidth=2, xscale=:log10, yscale=:log10,
    xlabel="ℓ", ylabel="absolute relative residual [%]",
    title="$(arguments["spectrum"]) emulator validation",
    legend=:topleft, framestyle=:box,
)
plot!(ell, residuals[2, :]; label="95%", linewidth=2)
plot!(ell, residuals[3, :]; label="99%", linewidth=2)
savefig(abspath(arguments["output"]))
println("Wrote validation plot: $(abspath(arguments["output"]))")
