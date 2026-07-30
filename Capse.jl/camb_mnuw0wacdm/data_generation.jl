ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "JULIA_PYTHONCALL_EXE", something(Sys.which("python"), "python"))

using Dates, Distributed, EmulatorsTrainer, JSON3, PythonCall, Random

const PARAMETER_NAMES = ["ln10As", "ns", "tau", "H0", "omega_b", "omega_c", "Mnu", "w0", "wa"]
const LOWER_BOUNDS = [2.5, 0.85, 0.02, 50.0, 0.02, 0.08, 0.0, -3.0, -3.0]
const UPPER_BOUNDS = [3.5, 1.05, 0.15, 90.0, 0.025, 0.16, 0.5, 1.0, 2.0]
const DESIGN_SEED = 20260735
const CAMB_DIRECTORY = @__DIR__

function get_option_int(name, default)
    prefix = name * "="
    for (index, argument) in enumerate(ARGS)
        startswith(argument, prefix) && return parse(Int, split(argument, "="; limit=2)[2])
        argument == name && index < length(ARGS) && return parse(Int, ARGS[index + 1])
    end
    return default
end

function positional_arguments()
    result = String[]
    skip = false
    for argument in ARGS
        if skip
            skip = false
        elseif startswith(argument, "--processes") && !occursin("=", argument)
            skip = true
        elseif !startswith(argument, "--")
            push!(result, argument)
        end
    end
    return result
end

n_processes = get_option_int("--processes", 2)
n_processes >= 0 || error("--processes must be non-negative")
positionals = positional_arguments()
n_samples = isempty(positionals) ? 500 : parse(Int, positionals[1])
output_directory = length(positionals) >= 2 ? abspath(positionals[2]) :
    joinpath(@__DIR__, "data", "camb_lcdm_$(n_samples)")
force = "--force" in ARGS

if n_processes > 0
    project = Base.active_project()
    addprocs(n_processes; exeflags="--project=$project")
end

@everywhere begin
    using EmulatorsTrainer, PythonCall
    const CAMB_WORKER = let
        sys = pyimport("sys")
        sys.path.insert(0, $CAMB_DIRECTORY)
        pyimport("camb_worker")
    end

    function compute_camb_observables(parameters)
        result = CAMB_WORKER.compute_spectra(parameters, 9000)
        values = Tuple(pyconvert(Vector{Float64}, result[name]) for name in
            ("TT", "TE", "EE", "PP", "TT_dense", "TE_dense", "EE_dense", "PP_dense"))
        arrays = (TT=values[1], TE=values[2], EE=values[3], PP=values[4],
            TT_dense=values[5], TE_dense=values[6], EE_dense=values[7], PP_dense=values[8])
        all(x -> all(isfinite, x), arrays) || error("CAMB output contains NaN or Inf")
        any(arrays.TT_dense .< 0) && error("TT contains negative values")
        any(arrays.EE_dense .< 0) && error("EE contains negative values")
        any(arrays.PP_dense .<= 0) && error("PP contains non-positive values")
        return arrays
    end
end

Random.seed!(DESIGN_SEED)
design = create_training_dataset(n_samples, LOWER_BOUNDS, UPPER_BOUNDS)
w0 = view(design, 8, :)
wa = copy(view(design, 9, :))
for (i, j) in zip(sortperm(w0), sortperm(wa; rev=true))
    design[9, i] = wa[j]
end
all(design[8, :] .+ design[9, :] .< 0) || error("w0+wa constraint failed")
static_axes = (
    ell_dense=collect(2.0:9000.0),
    ell_256=pyconvert(Vector{Float64}, CAMB_WORKER.lobatto_nodes(256, 2.0, 9000.0)),
    ell_192=pyconvert(Vector{Float64}, CAMB_WORKER.lobatto_nodes(192, 2.0, 9000.0)),
)
start = time()
dataset_file = compute_dataset_hdf5(
    design, PARAMETER_NAMES, output_directory, compute_camb_observables;
    mode=(n_processes > 0 ? :distributed : :serial),
    static_arrays=static_axes,
    force,
)
metadata = Dict(
    "created_at" => string(now()),
    "requested_samples" => n_samples,
    "mode" => (n_processes > 0 ? "distributed" : "serial"),
    "processes" => nworkers(),
    "dataset_file" => dataset_file,
    "parameter_names" => PARAMETER_NAMES,
    "lower_bounds" => LOWER_BOUNDS,
    "upper_bounds" => UPPER_BOUNDS,
    "design_seed" => DESIGN_SEED,
    "runtime_seconds" => time() - start,
    "constraint" => "w0 + wa < 0",
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
