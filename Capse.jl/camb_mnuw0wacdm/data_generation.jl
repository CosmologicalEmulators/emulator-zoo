using Dates, Distributed, EmulatorsTrainer, JSON3, PyCall

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .CapseCambMnuW0WaGeneration

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
    joinpath(@__DIR__, "data", "camb_mnuw0wacdm_$(n_samples)")
force = "--force" in ARGS

if n_processes > 0
    project = Base.active_project()
    addprocs(n_processes; exeflags=`--startup-file=no --project=$project`)
end

@everywhere begin
    using EmulatorsTrainer, PyCall
    include($GENERATION_FILE)
    using .CapseCambMnuW0WaGeneration
    const CAMB_BACKEND = CapseCambMnuW0WaGeneration.initialize_backend()
    function compute_camb_observables(parameters)
        return CapseCambMnuW0WaGeneration.compute_observables(parameters, CAMB_BACKEND)
    end
end

design = CapseCambMnuW0WaGeneration.create_design(n_samples)
master_backend = CapseCambMnuW0WaGeneration.initialize_backend()
start = time()
dataset_file = compute_dataset_hdf5(
    design, CapseCambMnuW0WaGeneration.PARAMETER_NAMES, output_directory, compute_camb_observables;
    mode=(n_processes > 0 ? :distributed : :serial),
    static_arrays=CapseCambMnuW0WaGeneration.static_axes(master_backend),
    force,
)
metadata = Dict(
    "created_at" => string(now()), "requested_samples" => n_samples,
    "mode" => (n_processes > 0 ? "distributed" : "serial"),
    "processes" => nworkers(), "dataset_file" => dataset_file,
    "parameter_names" => CapseCambMnuW0WaGeneration.PARAMETER_NAMES,
    "lower_bounds" => CapseCambMnuW0WaGeneration.LOWER_BOUNDS,
    "upper_bounds" => CapseCambMnuW0WaGeneration.UPPER_BOUNDS,
    "design_seed" => CapseCambMnuW0WaGeneration.DESIGN_SEED,
    "runtime_seconds" => time() - start, "constraint" => "w0 + wa < 0",
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
