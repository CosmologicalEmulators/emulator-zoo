using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using ArgParse
using EmulatorsTrainer
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--spectrum"
        help = "The Spectrum we are training. Either TT, TE, EE, or PP"
        default = "TT"
        "--path_input", "-i"
        help = "input folder"
        required = true
        "--path_output", "-o"
        help = "output folder"
        required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
global SpectraKind = parsed_args["spectra"]
CℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]

@info SpectraKind
@info CℓDirectory
@info OutDirectory

global nk = 2999

preprocess(ln10As, ns, H0, ombh2, omch2, τ) = exp(ln10As) * 1e-10

function get_observable_tuple(cosmo_pars, Cl)
    ombh2 = cosmo_pars["ombh2"]
    omch2 = cosmo_pars["omch2"]
    τ = cosmo_pars["τ"]
    H0 = cosmo_pars["H0"]
    ln10As = cosmo_pars["ln10As"]
    ns = cosmo_pars["ns"]

    factor = preprocess(ln10As, ns, H0, ombh2, omch2, τ)

    return (ln10As, ns, H0, ombh2, omch2, τ, Cl[1:2999] ./ factor)
end

n_input_features = 6
n_output_features = 2999
observable_file = "/" * SpectraKind * ".npy"
param_file = "/capse_dict.json"
add_observable!(df, location) = EmulatorsTrainer.add_observable_df!(df, location, param_file, observable_file, get_observable_tuple)

df = DataFrame(ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[], omega_cdm=Float64[], τ=Float64[], observable=Array[])
@time EmulatorsTrainer.load_df_directory!(df, CℓDirectory, add_observable!)

array_pars_in = ["ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "τ"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df, n_input_features, n_output_features)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array, n_output_features);

folder_output = OutDirectory * "/" * string(SpectraKind)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)

println(minimum(df[!, "ln10A_s"]), " , ", maximum(df[!, "ln10A_s"]))
println(minimum(df[!, "ns"]), " , ", maximum(df[!, "ns"]))
println(minimum(df[!, "H0"]), " , ", maximum(df[!, "H0"]))
println(minimum(df[!, "omega_b"]), " , ", maximum(df[!, "omega_b"]))
println(minimum(df[!, "omega_cdm"]), " , ", maximum(df[!, "omega_cdm"]))
println(minimum(df[!, "τ"]), " , ", maximum(df[!, "τ"]))
println(minimum(minimum(df[!, "observable"])), " , ", maximum(maximum(df[!, "observable"])))

NN_dict = JSON.parsefile("nn_setup.json")
NN_dict["n_output_features"] = n_output_features
NN_dict["n_input_features"] = n_input_features
mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict);

X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df, n_input_features, n_output_features);

p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))

l = Array(2:3000)
dest = joinpath(folder_output, "l.npy")  # constructs the full destination path nicely
npzwrite(dest, l)

dest = joinpath(folder_output, "nn_setup.json")
json_str = JSON.json(NN_dict)
open(dest, "w") do file
    write(file, json_str)
end


dest = joinpath(folder_output, "postprocessing.py")
run(`cp postprocessing.py $dest`)
dest = joinpath(folder_output, "postprocessing.jl")
run(`cp postprocessing.jl $dest`)


report = let mtrain = mlpdloss, X = X, Xtest = Xtest, mtest = mlpdtest
    p -> begin
        let train = mlpdloss(X, p), test = mlpdtest(Xtest, p)
            @info "Loss:" train test
        end
    end
end;

pippo_loss = mlpdtest(Xtest, p)
println("Initial Loss: ", pippo_loss)
lr_list = [1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7]

for lr in lr_list
    for i in 1:10
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), 1000
            ; batchsize=128)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p)
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end
