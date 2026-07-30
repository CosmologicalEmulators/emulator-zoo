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
        "--spectrum"
        help = "The Spectrum we are training. Either TT, TE, EE, or PP"
        default = "TT"
        "--path_input", "-i"
        help = "merged HDF5 dataset file"
        required = true
        "--path_output", "-o"
        help = "output folder"
        required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
global SpectraKind = parsed_args["spectrum"]
CℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]

@info SpectraKind
@info CℓDirectory
@info OutDirectory


preprocess(ln10As, ns, H0, ombh2, omch2, τ, Mν, w0, wa) = exp(ln10As) * 1e-10 * exp(-2 * τ)

function get_observable_tuple(cosmo_pars, Cl)
    ombh2 = cosmo_pars["ombh2"]
    omch2 = cosmo_pars["omch2"]
    τ = cosmo_pars["τ"]
    H0 = cosmo_pars["H0"]
    ln10As = cosmo_pars["ln10As"]
    ns = cosmo_pars["ns"]
    Mν = cosmo_pars["Mν"]
    w0 = cosmo_pars["w0"]
    wa = cosmo_pars["wa"]

    factor = preprocess(ln10As, ns, H0, ombh2, omch2, τ, Mν, w0, wa)

    return (ln10As, ns, H0, ombh2, omch2, τ, Mν, w0, wa, Cl[1:9998] ./ factor)
end

n_input_features = 9
n_output_features = 9998
dataset = EmulatorsTrainer.load_hdf5_dataset(CℓDirectory)
parameters = dataset.parameters
parameter_names = dataset.parameter_names
observable = get(dataset.observables, Symbol(SpectraKind), nothing)
observable === nothing && error("Observable $SpectraKind is not present in $CℓDirectory")
all(dataset.valid) || error("HDF5 dataset contains invalid samples")

# Keep the existing trainer interface: construct the DataFrame from the
# merged arrays, without reopening one file per cosmology.
df = DataFrame(ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[],
    omega_cdm=Float64[], τ=Float64[], Mν=Float64[], w0=Float64[], wa=Float64[], observable=Array[])
for sample_index in axes(parameters, 1)
    cosmo_pars = Dict(parameter_names[j] => parameters[sample_index, j] for j in axes(parameters, 2))
    push!(df, get_observable_tuple(cosmo_pars, observable[sample_index, :]))
end

array_pars_in = ["ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "τ", "Mν", "w0", "wa"]
_, out_array = EmulatorsTrainer.extract_input_output_df(df; input_columns=Symbol.(array_pars_in))
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array);

folder_output = OutDirectory * "/" * string(SpectraKind)
mkpath(folder_output)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax; input_columns=Symbol.(array_pars_in))

println(minimum(df[!, "ln10A_s"]), " , ", maximum(df[!, "ln10A_s"]))
println(minimum(df[!, "ns"]), " , ", maximum(df[!, "ns"]))
println(minimum(df[!, "H0"]), " , ", maximum(df[!, "H0"]))
println(minimum(df[!, "omega_b"]), " , ", maximum(df[!, "omega_b"]))
println(minimum(df[!, "omega_cdm"]), " , ", maximum(df[!, "omega_cdm"]))
println(minimum(df[!, "τ"]), " , ", maximum(df[!, "τ"]))
println(minimum(df[!, "Mν"]), " , ", maximum(df[!, "Mν"]))
println(minimum(df[!, "w0"]), " , ", maximum(df[!, "w0"]))
println(minimum(df[!, "wa"]), " , ", maximum(df[!, "wa"]))
println(minimum(minimum(df[!, "observable"])), " , ", maximum(maximum(df[!, "observable"])))

NN_dict = JSON.parsefile(joinpath(@__DIR__, "nn_setup.json"))
NN_dict["n_output_features"] = n_output_features
NN_dict["n_input_features"] = n_input_features
NN_dict["emulator_description"] = Dict("source"=>"CLASS Mnu-w0-wa", "spectrum"=>SpectraKind)
mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict);

X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df; input_columns=Symbol.(array_pars_in), seed=20260753);

p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))

l = Array(2:9999)
dest = joinpath(folder_output, "l.npy")  # constructs the full destination path nicely
npzwrite(dest, l)

dest = joinpath(folder_output, "nn_setup.json")
json_str = JSON.json(NN_dict)
open(dest, "w") do file
    write(file, json_str)
end


dest = joinpath(folder_output, "postprocessing.py")
run(`cp $(joinpath(@__DIR__, "postprocessing.py")) $dest`)
dest = joinpath(folder_output, "postprocessing.jl")
run(`cp $(joinpath(@__DIR__, "postprocessing.jl")) $dest`)


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

steps_per_session = parse(Int, get(ENV, "CAPSE_STEPS_PER_SESSION", "1000"))
sessions_per_rate = parse(Int, get(ENV, "CAPSE_SESSIONS_PER_RATE", "10"))
batch_size = parse(Int, get(ENV, "CAPSE_BATCH_SIZE", "128"))
for lr in lr_list
    for i in 1:sessions_per_rate
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), steps_per_session
            ; batchsize=batch_size)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p)
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end
