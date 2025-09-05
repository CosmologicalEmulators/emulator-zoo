using Pkg
Pkg.activate(".")

using Statistics, Plots, LinearAlgebra, NPZ, Random
using LaTeXStrings
using NPZ
using Capse
using EmulatorsTrainer

pars_array = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mν", "w0", "wa"]
Cℓ_directory = "/home/mbonici/Desktop/claude-test/capse_class_mnuw0wacdm_40000"


ℓ = Array(2:3000);

weight_master_folder = "/home/mbonici/Desktop/phalanx_test/trained_capse_mnuw0wacdm_40000_6/"

weights_folder = weight_master_folder * "/TT"

CℓTT_emu = Capse.load_emulator(weights_folder);


ℓ = CℓTT_emu.ℓgrid


get_emu_prediction(p) = Capse.get_Cℓ(p, CℓTT_emu)
get_ground_truth(location) = npzread(location * "/TT.npy")[3:3001]
sorted_residuals_TT = EmulatorsTrainer.evaluate_sorted_residuals(Cℓ_directory, "capse_dict.json", pars_array,
    get_ground_truth, get_emu_prediction)

npzwrite("sorted_residuals_TT.npy", sorted_residuals_TT)

plt = Plots.plot(sorted_residuals_TT[1, :], xlabel=L"ℓ", ylabel=L"\frac{\left|\Delta C^{TT}\right|}{\sigma_{{TT}}}(\ell)\quad \mathrm{distribution}", legend=:topleft, label=L"68\%")
Plots.plot!(plt, sorted_residuals_TT[2, :], label=L"95\%")
Plots.plot!(plt, sorted_residuals_TT[3, :], label=L"99\%")
savefig("error_distribution_TT.png")
savefig("error_distribution_TT.pdf")


weights_folder = weight_master_folder * "/EE"
CℓEE_emu = Capse.load_emulator(weights_folder)

get_emu_prediction(p) = Capse.get_Cℓ(p, CℓEE_emu)
get_ground_truth(location) = npzread(location * "/EE.npy")[3:3001]
sorted_residuals_EE = EmulatorsTrainer.evaluate_sorted_residuals(Cℓ_directory, "capse_dict.json", pars_array,
    get_ground_truth, get_emu_prediction)

npzwrite("sorted_residuals_EE.npy", sorted_residuals_EE)

plt = Plots.plot(sorted_residuals_EE[1, :], xlabel=L"ℓ", ylabel=L"\frac{\left|\Delta C^{EE}\right|}{\sigma_{{EE}}}(\ell)\quad \mathrm{distribution}", legend=:topleft, label=L"68\%")
Plots.plot!(plt, sorted_residuals_EE[2, :], label=L"95\%")
savefig("error_distribution_EE.png")
savefig("error_distribution_EE.pdf")

weights_folder = weight_master_folder * "/TE"
CℓTE_emu = Capse.load_emulator(weights_folder)

get_emu_prediction(p) = Capse.get_Cℓ(p, CℓTE_emu)
get_ground_truth(location) = npzread(location * "/TE.npy")[3:3001]
sorted_residuals_TE = EmulatorsTrainer.evaluate_sorted_residuals(Cℓ_directory, "capse_dict.json", pars_array,
    get_ground_truth, get_emu_prediction)

npzwrite("sorted_residuals_TE.npy", sorted_residuals_TE)

plt = Plots.plot(sorted_residuals_TE[1, :], xlabel=L"ℓ", ylabel=L"\frac{\left|\Delta C^{TE}\right|}{\sigma_{{TE}}}(\ell)\quad \mathrm{distribution}", legend=:topleft, label=L"68\%")
Plots.plot!(plt, sorted_residuals_TE[2, :], label=L"95\%")
#Plots.plot!(plt, sorted_residuals_TT[3,:], label = L"99\%")
savefig("error_distribution_TE.png")
savefig("error_distribution_TE.pdf")


weights_folder = weight_master_folder * "/PP"
CℓPP_emu = Capse.load_emulator(weights_folder)

get_emu_prediction(p) = Capse.get_Cℓ(p, CℓPP_emu)
get_ground_truth(location) = npzread(location * "/PP.npy")[3:3001]
sorted_residuals_PP = EmulatorsTrainer.evaluate_sorted_residuals(Cℓ_directory, "capse_dict.json", pars_array,
    get_ground_truth, get_emu_prediction)

npzwrite("sorted_residuals_PP.npy", sorted_residuals_PP)

plt = Plots.plot(sorted_residuals_PP[1, :], xlabel=L"ℓ", ylabel=L"\frac{\left|\Delta C^{PP}\right|}{\sigma_{{PP}}}(\ell)\quad \mathrm{distribution}", legend=:topleft, label=L"68\%")
Plots.plot!(plt, sorted_residuals_PP[2, :], label=L"95\%")
#Plots.plot!(plt, sorted_residuals_TT[3,:], label = L"99\%")
savefig("error_distribution_PP.png")
savefig("error_distribution_PP.pdf")
