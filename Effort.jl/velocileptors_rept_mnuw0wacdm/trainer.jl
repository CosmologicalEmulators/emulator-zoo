using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using Effort
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
global nk = 80

if Componentkind == "11"
    nk_factor = 3
elseif Componentkind == "loop"
    nk_factor = 9
elseif Componentkind == "ct"
    nk_factor = 4
else
    @error "Wrong component!"
end

function reshape_Pk(Pk, factor)
    if Componentkind == "11"
        result = vec(Array(Pk)[:, 1:3]) ./ factor
    elseif Componentkind == "loop"
        result = vec(Array(Pk)[:, 4:12]) ./ factor^2
    elseif Componentkind == "ct"
        result = vec(Array(Pk)[:, 13:16]) ./ factor
    else
        @error "Wrong component!"
    end
    return result
end

function D_ODE(z, ωb, ωcdm, h, Mν, w0, wa)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=3.0, nₛ=0.96, h=h,
        ωb=ωb, ωc=ωcdm, mν=Mν,
        w0=w0, wa=wa
    )

    return Effort.D_z(z, cosmology)
end

preprocess(z, As, ωb, ωcdm, h, Mν, w0, wa) = As * D_ODE(z, ωb, ωcdm, h, Mν, w0, wa)^2

function get_observable_tuple(cosmo_pars, Pk)
    z = cosmo_pars["z"]
    ωb = cosmo_pars["ombh2"]
    ωcdm = cosmo_pars["omch2"]
    Mν = cosmo_pars["Mν"]
    h = cosmo_pars["H0"] / 100
    As = exp(cosmo_pars["ln10As"]) * 1e-10
    w0 = cosmo_pars["w0"]
    wa = cosmo_pars["wa"]

    factor = preprocess(z, As, ωb, ωcdm, h, Mν, w0, wa)
    return (cosmo_pars["z"], cosmo_pars["ln10As"], cosmo_pars["ns"], cosmo_pars["H0"],
        cosmo_pars["ombh2"], cosmo_pars["omch2"], cosmo_pars["Mν"], cosmo_pars["w0"], cosmo_pars["wa"], reshape_Pk(Pk, factor))
end

n_input_features = 9
n_output_features = nk * nk_factor

observable_name = Symbol("pk_$(ℓ)")
dataset = EmulatorsTrainer.load_hdf5_dataset(PℓDirectory)
parameters_array = dataset.parameters
parameter_names = dataset.parameter_names
observable = get(dataset.observables, observable_name, nothing)
observable === nothing && error("Observable $observable_name is not present in $PℓDirectory")
all(dataset.valid) || error("HDF5 dataset contains invalid samples")

df = DataFrame(z=Float64[], ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[], omega_cdm=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[], observable=Array[])
for sample_index in axes(parameters_array, 1)
    cosmo_pars = Dict(parameter_names[j] => parameters_array[sample_index, j]
        for j in axes(parameters_array, 2))
    push!(df, get_observable_tuple(cosmo_pars, observable[sample_index, :, :]))
end
size(df, 1) >= 2 || error("Too few samples")

array_pars_in = ["z", "ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "Mν", "w0", "wa"]
_, out_array = EmulatorsTrainer.extract_input_output_df(df; input_columns=Symbol.(array_pars_in))
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array);

folder_output = OutDirectory * "/" * string(ℓ) * "/" * string(Componentkind)
mkpath(folder_output)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax; input_columns=Symbol.(array_pars_in))

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

NN_dict = Dict{String,Any}(
    "n_input_features" => n_input_features,
    "n_output_features" => n_output_features,
    "n_hidden_layers" => 5,
    "emulator_description" => Dict(
        "source" => "CLASS + Velocileptors REPT",
        "cosmology" => "Mnu-w0-waCDM",
        "component" => Componentkind,
        "multipole" => ℓ,
    ),
    "layers" => Dict(
        "layer_$index" => Dict("n_neurons" => 64, "activation_function" => "tanh")
        for index in 1:5
    ),
)
mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict);

X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df; input_columns=Symbol.(array_pars_in), seed=20260745);

p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd);

npzwrite(joinpath(folder_output, "k.npy"), vec(dataset.observables[:kv][1, :]))

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

if ℓ == 0
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_0.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_0.jl $dest`)
elseif ℓ == 2
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_2.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_2.jl $dest`)
elseif ℓ == 4
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_4.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_4.jl $dest`)
else
    @error "Unsupported multipole"
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
lr_list = [1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7, 5e-6, 2e-7]

steps_per_session = parse(Int, get(ENV, "CAPSE_STEPS_PER_SESSION", "2000"))
sessions_per_rate = parse(Int, get(ENV, "CAPSE_SESSIONS_PER_RATE", "20"))
batch_size = parse(Int, get(ENV, "CAPSE_BATCH_SIZE", "128"))

for lr in lr_list
    for i in 1:sessions_per_rate
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), 2000
            ; batchsize=512)
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
