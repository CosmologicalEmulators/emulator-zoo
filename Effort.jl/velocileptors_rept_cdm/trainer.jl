using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using ArgParse
using DelimitedFiles

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
@info ℓ
@info PℓDirectory
@info OutDirectory
@info Componentkind
global nk = 50

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
    if Componentkind == "11"
        result = vec(Array(Pk)[:, 1:3]) ./ As
    elseif Componentkind == "loop"
        result = vec(Array(Pk)[:, 4:12]) ./ As^2
    elseif Componentkind == "ct"
        result = vec(Array(Pk)[:, 13:16]) ./ As
    elseif Componentkind == "st"
        result = vec(Array(Pk)[:, 17:19])
    else
        @error "Wrong component!"
    end
    return result
end

function get_observable_tuple(cosmo_pars, Pk)
    As = exp(cosmo_pars["ln10As"]) * 1e-10
    return (cosmo_pars["ln10As"], cosmo_pars["H0"], cosmo_pars["omch2"], reshape_Pk(Pk, As))
end

n_input_features = 3
n_output_features = nk * nk_factor

observable_name = Symbol("pk_$(ℓ)")
dataset = EmulatorsTrainer.load_hdf5_dataset(PℓDirectory)
parameters_array = dataset.parameters
parameter_names = dataset.parameter_names
observable = get(dataset.observables, observable_name, nothing)
observable === nothing && error("Observable $observable_name is not present in $PℓDirectory")
all(dataset.valid) || error("HDF5 dataset contains invalid samples")

df = DataFrame(ln10As=Float64[], H0=Float64[], omch2=Float64[], observable=Array[])
for sample_index in axes(parameters_array, 1)
    cosmo_pars = Dict(parameter_names[j] => parameters_array[sample_index, j]
        for j in axes(parameters_array, 2))
    push!(df, get_observable_tuple(cosmo_pars, observable[sample_index, :, :]))
end
size(df, 1) >= 2 || error("Too few samples")

array_pars_in = ["ln10As", "H0", "omch2"]
_, out_array = EmulatorsTrainer.extract_input_output_df(df; input_columns=Symbol.(array_pars_in))
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array);

folder_output = OutDirectory * "/" * string(ℓ) * "/" * string(Componentkind)
mkpath(folder_output)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax; input_columns=Symbol.(array_pars_in))

println(minimum(df[!, "ln10As"]), " , ", maximum(df[!, "ln10As"]))
println(minimum(df[!, "H0"]), " , ", maximum(df[!, "H0"]))
println(minimum(df[!, "omch2"]), " , ", maximum(df[!, "omch2"]))
println(minimum(minimum(df[!, "observable"])), " , ", maximum(maximum(df[!, "observable"])))

NN_dict = Dict{String,Any}("n_input_features"=>n_input_features,"n_output_features"=>n_output_features,"n_hidden_layers"=>5,"emulator_description"=>Dict("source"=>"CLASS + Velocileptors REPT","cosmology"=>"CDM fixed-z","component"=>Componentkind,"multipole"=>ℓ),"layers"=>Dict("layer_$i"=>Dict("n_neurons"=>64,"activation_function"=>"tanh") for i in 1:5))
mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict);

X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df; input_columns=Symbol.(array_pars_in), seed=20260749);

p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd);

k = readdlm("k.txt", ' ')[:,1]
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
elseif Componentkind == "st"
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing_st.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing_st.jl $dest`)
else
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing.jl $dest`)
end

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
steps_per_session = parse(Int, get(ENV, "CAPSE_STEPS_PER_SESSION", "2000"))
sessions_per_rate = parse(Int, get(ENV, "CAPSE_SESSIONS_PER_RATE", "10"))
batch_size = parse(Int, get(ENV, "CAPSE_BATCH_SIZE", "128"))

for lr in lr_list
    for i in 1:sessions_per_rate
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), 2000   #η = 1e-4
            ; batchsize=256)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p)
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end



exit()
