using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--component"
            help = "the component we are training. Either 11, loop, ct or st"
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
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
global Componentkind = parsed_args["component"]
ℓ = parsed_args["multipole"]
PℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]

global nk = 50

global Componentkind = "11"
ℓ = 0
PℓDirectory = "/farmdisk1/mbonici/effort_velocileptors_1000"
OutDirectory = "/farmdisk1/mbonici/trained_effort_velocileptors_1000_lcdm"

if Componentkind == "11"
    nk_factor = 3
elseif Componentkind == "loop"
    nk_factor = 9
elseif Componentkind == "ct"
    nk_factor = 4
elseif Componentkind == "st"
    nk_factor = 3
else
    @error "Wrong component!"
end

function reshape_Pk(Pk, As)
    @info "Pippo1"
    if Componentkind == "11"
        result = vec(Array(Pk)[:,1:3])./As
    elseif Componentkind == "loop"
        result = vec(Array(Pk)[:,4:12])./As^2
    elseif Componentkind == "ct"
        result = vec(Array(Pk)[:,13:16])./As
    elseif Componentkind == "st"
        result = vec(Array(Pk)[:,17:19])
    else
        @error "Wrong component!"
    end
    return result
end

function get_observable_tuple(cosmo_pars, Pk)
    @info "Pippo2!"
    As = exp(cosmo_pars["ln10As"])*1e-10
    return (cosmo_pars["ln10As"], cosmo_pars["H0"], cosmo_pars["omch2"], reshape_Pk(Pk, As))
end

n_input_features = 3
n_output_features = nk*nk_factor
Pkkind = "0"
observable_file = "/pk_"*string(ℓ)*".npy"
param_file = "/effort_dict.json"
add_observable!(df, location) = EmulatorsTrainer.add_observable_df!(df, location, param_file, observable_file, get_observable_tuple)

df = DataFrame(ln10As = Float64[], H0 = Float64[], omch2 = Float64[], observable = Array[])
@info PℓDirectory
@time EmulatorsTrainer.load_df_directory!(df, PℓDirectory, add_observable!)


array_pars_in = ["ln10As", "H0", "omch2"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df, n_input_features, n_output_features)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array, n_output_features);

folder_output = OutDirectory*"/"string(ℓ)*"/"*string(Componentkind)
npzwrite(folder_output*"/inminmax.npy", in_MinMax)
npzwrite(folder_output*"/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)

println(minimum(df[!,"ln10As"]), " , ", maximum(df[!,"ln10As"]))
println(minimum(df[!,"H0"]), " , ", maximum(df[!,"H0"]))
println(minimum(df[!,"omch2"]), " , ", maximum(df[!,"omch2"]))
println(minimum(minimum(df[!,"observable"])), " , ", maximum(maximum(df[!,"observable"])))

NN_dict = JSON.parsefile("nn_setup.json")
NN_dict["n_output_features"] = n_output_features
NN_dict["n_input_features"] = n_input_features
mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict);

X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df, n_input_features, n_output_features);

p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))

report = let mtrain = mlpdloss, X=X, Xtest=Xtest, mtest = mlpdtest
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
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), 2000   #η = 1e-4
                                                ; batchsize =256);
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output*"/weights.npy", p)
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end
