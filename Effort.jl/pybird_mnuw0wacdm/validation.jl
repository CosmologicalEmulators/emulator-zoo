using DataFrames
using Plots
using BenchmarkTools
using NPZ
using LaTeXStrings
using Effort
using Statistics
using EmulatorsTrainer

D_prec(z, Ωcb0, h, Mν, w0, wa) = Effort._D_z(z, Ωcb0, h; mν=Mν, w0=w0, wa=wa)

function effort_load(root)
    P0 = Effort.load_multipole_emulator(root * "/0/")
    P2 = Effort.load_multipole_emulator(root * "/2/")
    P4 = Effort.load_multipole_emulator(root * "/4/")
    return P0, P2, P4
end

pars_array = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]

function get_ground_truth(location, lidx)
    P11l = npzread(location * "/P11l.npy")[lidx, :, :]
    Ploopl = npzread(location * "/Ploopl.npy")[lidx, :, :]
    Pctl = npzread(location * "/Pctl.npy")[lidx, :, :]
    b1, b2, b3, b4, b5, b6, b7, f = biases
    mybiases = Array([b1^2, 2 * b1 * f, f^2, 1.0, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3,
        b1 * b4, b2 * b2, b2 * b4, b4 * b4, 2 * b1 * b5, 2 * b1 * b6, 2 * b1 * b7, 2 * f * b5, 2 * f * b6, 2 * f * b7])
    return hcat(P11l', Ploopl', Pctl') * mybiases
end

function get_emu_prediction(input_test, Pℓ_emu, D::Function)
    z, ln10As, ns, H0, ombh2, omch2, Mν, w0, wa = input_test
    h = H0 / 100
    Ωcb0 = (ombh2 + omch2) / h^2
    myD = D(z, Ωcb0, h, Mν, w0, wa)
    Effort.get_Pℓ(input_test, myD, biases, Pℓ_emu)
end

P0, P2, P4 = effort_load("trained_effort_pybird_mnuw0wacdm_big_nn_60000/")
k = P0.P11.kgrid

Pℓ_directory = "/home/marcobonici/Desktop/work/CosmologicalEmulators/emulator-zoo/Effort.jl/pybird_mnuw0wacdm/effort_pybird_10000_mnuw0wacdm"

#get_ground_truth(location) = get_ground_truth(location, lidx)
#get_emu_prediction(input_test) = get_emu_prediction(input_test, P0, D_prec)

D = 1.0
biases = [2, 1, -1.0, 1, -2, -1.0, 0.0, 0.86]

#validation_residuals = EmulatorsTrainer.evaluate_sorted_residuals(Pℓ_directory, "effort_dict.json", pars_array,
#    get_ground_truth, get_emu_prediction, 10000, 74)

function validation(lidx, Pl)
    gt(location) = get_ground_truth(location, lidx)
    get_pred(input_test) = get_emu_prediction(input_test, Pl, D_prec)

    D = 1.0
    biases = [2, 1, -1.0, 1, -2, -1.0, 0.0, 0.86]

    validation_residuals = EmulatorsTrainer.evaluate_sorted_residuals(Pℓ_directory, "effort_dict.json", pars_array,
        gt, get_pred, 10000, 74)
end

res_0 = validation(1, P0)
res_2 = validation(2, P2)
res_4 = validation(3, P4)

plot_font = "Computer Modern"

Plots.default(titlefont=(16, plot_font), fontfamily=plot_font,
    linewidth=2, framestyle=:box, fg_legend=:black, label=nothing, grid=false, tickfontsize=12, size=(550, 400), labelfontsize=13, dpi=200, minorgrid=true)#, xticks = [1,10,100, 1000])

p0 = plot(ylabel=L"\frac{\Delta P_0}{P_0}\%",
    size=(1000, 600),
    bottom_margin=-6Plots.mm,
    left_margin=15Plots.mm,
    xlabelfontsize=18,
    ylabelfontsize=18,
    titlefontsize=20,
    xtickfont=2,
    ytickfont=16,
    legendfontsize=15,
    axislinewidth=2, title="95% distributions")

plot!(p0, k, res_0[1, :], color="orange", label="120000, batch 128, big NN")
plot!(p0, k, res_0[2, :], color="orange", label="")
#plot!(p0, k, res_0[3, :], color="orange", label="")

p2 = plot(ylabel=L"\frac{\Delta P_2}{P_2}\%",
    size=(1000, 600),
    bottom_margin=-6Plots.mm,
    top_margin=-2.5Plots.mm,
    left_margin=15Plots.mm,
    xlabelfontsize=18,
    ylabelfontsize=18,
    titlefontsize=20,
    xtickfont=2,
    ytickfont=16,
    legendfontsize=15)


plot!(p2, k, res_2[1, :], color="orange", label="")
plot!(p2, k, res_2[2, :], color="orange", label="")
#plot!(p2, k, res_2[3, :], color="orange", label="")

p4 = plot(xlabel=L"k[h/\mathrm{Mpc}]", ylabel=L"\frac{\Delta P_4}{P_4}\%", size=(1000, 500), left_margin=10Plots.mm, bottom_margin=6Plots.mm, xlabelfontsize=18, # Set x-label font size
    ylabelfontsize=18, # Set y-label font size
    titlefontsize=20,
    top_margin=-2.5Plots.mm,
    tickfont=16,
    legendfontsize=15)
plot!(p4, k, res_4[1, :], color="orange", label="")
plot!(p4, k, res_4[2, :], color="orange", label="")
#plot!(p4, k, res_4[3, :], color="orange", label="")


final_plot = plot(p0, p2, p4, layout=(3, 1), size=(1200, 1100), left_margin=5Plots.mm)
savefig("error_distribution_mnulcdm.png")
savefig("error_distribution_mnulcdm.pdf")
display(final_plot)
