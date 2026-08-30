using Dates, Distributed, EmulatorsTrainer, HDF5, JSON3, PyCall

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .CapseCambMnuOkGeneration

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
        elseif argument in ("--processes", "--seed")
            skip = true
        elseif !startswith(argument, "--")
            push!(result, argument)
        end
    end
    return result
end

n_processes = get_option_int("--processes", 2)
seed = get_option_int("--seed", CapseCambMnuOkGeneration.DESIGN_SEED)
n_processes >= 0 || error("--processes must be non-negative")
positionals = positional_arguments()
n_samples = isempty(positionals) ? 500 : parse(Int, positionals[1])
output_directory = length(positionals) >= 2 ? abspath(positionals[2]) :
    joinpath(@__DIR__, "data", "camb_mnuoklcdm_$(n_samples)")
force = "--force" in ARGS

if n_processes > 0
    project = Base.active_project()
    addprocs(n_processes; exeflags=`--startup-file=no --project=$project`)
end

@everywhere begin
    using EmulatorsTrainer, PyCall
    include($GENERATION_FILE)
    using .CapseCambMnuOkGeneration
    const CAMB_BACKEND = Ref{Any}(nothing)
    function compute_camb_observables(parameters)
        backend = CAMB_BACKEND[]
        if backend === nothing
            backend = CapseCambMnuOkGeneration.initialize_backend()
            CAMB_BACKEND[] = backend
        end
        return CapseCambMnuOkGeneration.compute_observables(parameters, backend)
    end
end

design = CapseCambMnuOkGeneration.create_design(n_samples; seed)
master_backend = CapseCambMnuOkGeneration.initialize_backend()
start = time()
dataset_file = compute_dataset_hdf5(
    design, CapseCambMnuOkGeneration.PARAMETER_NAMES, output_directory, compute_camb_observables;
    mode=(n_processes > 0 ? :distributed : :serial),
    static_arrays=CapseCambMnuOkGeneration.static_axes(master_backend),
    force, skip_errors=true,
)
retained_samples = h5open(dataset_file, "r") do file
    length(file["sample_indices"])
end
failure_file = joinpath(output_directory, "generation_failures.json")
failures = isfile(failure_file) ? JSON3.read(read(failure_file, String)) : []
omega_k = view(design, 8, :)
metadata = Dict(
    "created_at" => string(now()), "requested_samples" => n_samples,
    "mode" => (n_processes > 0 ? "distributed" : "serial"),
    "processes" => nworkers(), "dataset_file" => dataset_file,
    "parameter_names" => CapseCambMnuOkGeneration.PARAMETER_NAMES,
    "lower_bounds" => CapseCambMnuOkGeneration.LOWER_BOUNDS,
    "upper_bounds" => CapseCambMnuOkGeneration.UPPER_BOUNDS,
    "design_seed" => seed,
    "retained_samples" => retained_samples,
    "failed_samples" => length(failures),
    "OmegaK_min" => minimum(omega_k),
    "OmegaK_max" => maximum(omega_k),
    "camb_version" => string(master_backend.camb.__version__),
    "python_executable" => PyCall.python,
    "runtime_seconds" => time() - start,
    "model" => "Mnu-OmegaK-LambdaCDM with w0=-1 and wa=0",
    "camb_configuration" => CapseCambMnuOkGeneration.backend_configuration(master_backend),
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
