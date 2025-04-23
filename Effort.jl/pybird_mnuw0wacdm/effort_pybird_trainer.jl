using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using ArgParse
using EmulatorsTrainer
using Effort
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--component"
        help = "the component we are training. Either 11, loop, or ct"
        default = "11"
        "--multipole", "-l"
        help = "the multipole we are training. Either 0, 2, or 4"
        arg_type = Int
        default = 0
        "--path_input", "-i"
        help = "input folder"
        required = true
        "--path_output", "-o"
        help = "output folder"
        required = true
        "--preprocessing", "-p"
        help = "How to preprocess data"
        arg_type = String
        required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
global Componentkind = parsed_args["component"]
ℓ = parsed_args["multipole"]
PℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]
Preprocessing = parsed_args["preprocessing"]

@info Componentkind
@info ℓ
@info PℓDirectory
@info OutDirectory
@info Preprocessing

D_ODE(z, Ωcb0, h, Mν, w0, wa) = Effort._D_z(z, Ωcb0, h; mν=Mν, w0=w0, wa=wa)

global nk = 74

if Componentkind == "11"
    nk_factor = 3
elseif Componentkind == "loop"
    nk_factor = 12
elseif Componentkind == "ct"
    nk_factor = 6
else
    @error "Wrong component!"
end

if ℓ == 0
    ℓidx = 1
elseif ℓ == 2
    ℓidx = 2
elseif ℓ == 4
    ℓidx = 3
else
    @error "Wrong multipole"
end

if Preprocessing == "noprec"
    preprocess(z, As, Ωcb0, h, Mν, w0, wa) = 1
elseif Preprocessing == "Asprec"
    preprocess(z, As, Ωcb0, h, Mν, w0, wa) = As
elseif Preprocessing == "Dzprec"
    preprocess(z, As, Ωcb0, h, Mν, w0, wa) = D_ODE(z, Ωcb0, h, Mν, w0, wa)^2
elseif Preprocessing == "AsDzprec"
    preprocess(z, As, Ωcb0, h, Mν, w0, wa) = As * D_ODE(z, Ωcb0, h, Mν, w0, wa)^2
else
    @error "Wrong preprocessing"
end

function reshape_Pk(Pk, factor)
    if Componentkind == "11"
        result = vec((Array(Pk)[ℓidx, :, :])') ./ factor
    elseif Componentkind == "loop"
        result = vec((Array(Pk)[ℓidx, :, :])') ./ factor^2
    elseif Componentkind == "ct"
        result = vec((Array(Pk)[ℓidx, :, :])') ./ factor
    else
        @error "Wrong component!"
    end
    return result
end

function get_observable_tuple(cosmo_pars, Pk)
    z = cosmo_pars["z"]
    ombh2 = cosmo_pars["ombh2"]
    omch2 = cosmo_pars["omch2"]
    Mν = cosmo_pars["Mν"]
    h = cosmo_pars["H0"] / 100
    Ωcb0 = (ombh2 + omch2) / h^2
    As = exp(cosmo_pars["ln10As"]) * 1e-10
    w0 = cosmo_pars["w0"]
    wa = cosmo_pars["wa"]

    factor = preprocess(z, As, Ωcb0, h, Mν, w0, wa)

    return (cosmo_pars["z"], cosmo_pars["ln10As"], cosmo_pars["ns"], cosmo_pars["H0"],
        cosmo_pars["ombh2"], cosmo_pars["omch2"], cosmo_pars["Mν"], cosmo_pars["w0"], cosmo_pars["wa"], reshape_Pk(Pk, factor))
end

n_input_features = 9
n_output_features = nk * nk_factor
observable_file = "/P" * Componentkind * "l.npy"
param_file = "/effort_dict.json"
add_observable!(df, location) = EmulatorsTrainer.add_observable_df!(df, location, param_file, observable_file, get_observable_tuple)

df = DataFrame(z=Float64[], ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[], omega_cdm=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[], observable=Array[])
@time EmulatorsTrainer.load_df_directory!(df, PℓDirectory, add_observable!)

array_pars_in = ["z", "ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "Mν", "w0", "wa"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df, n_input_features, n_output_features)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array, n_output_features);

folder_output = OutDirectory * "/" * string(ℓ) * "/" * string(Componentkind)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)

println(minimum(df[!, "z"]), " , ", maximum(df[!, "z"]))
println(minimum(df[!, "ln10A_s"]), " , ", maximum(df[!, "ln10A_s"]))
println(minimum(df[!, "ns"]), " , ", maximum(df[!, "ns"]))
println(minimum(df[!, "H0"]), " , ", maximum(df[!, "H0"]))
println(minimum(df[!, "omega_b"]), " , ", maximum(df[!, "omega_b"]))
println(minimum(df[!, "omega_cdm"]), " , ", maximum(df[!, "omega_cdm"]))
println(minimum(df[!, "Mν"]), " , ", maximum(df[!, "Mν"]))
println(minimum(df[!, "w0"]), " , ", maximum(df[!, "w0"]))
println(minimum(df[!, "wa"]), " , ", maximum(df[!, "wa"]))
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
            ; batchsize=1024)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p)
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end
k = readdlm("k.txt", ' ')
dest = joinpath(folder_output, "k.npy")  # constructs the full destination path nicely
npzwrite(dest, k)

dest = joinpath(folder_output, "nn_setup.json")
json_str = JSON.json(NN_dict)
open(dest, "w") do file
    write(file, json_str)
end

if Componentkind == "loop"
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing_loop.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing_loop.jl $dest`)
else
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing.jl $dest`)
end
