ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "JULIA_PYTHONCALL_EXE", something(Sys.which("python"), "python"))

using Dates, Distributed, EmulatorsTrainer, JSON3, PythonCall, SlurmClusterManager

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .CapseCambMnuW0WaGeneration

function option(name, default)
    prefix = name * "="
    for (index, argument) in enumerate(ARGS)
        startswith(argument, prefix) && return split(argument, "="; limit=2)[2]
        argument == name && index < length(ARGS) && return ARGS[index + 1]
    end
    return default
end

n_samples = parse(Int, option("--samples", "20000"))
output_directory = abspath(option("--output", error("--output is required")))
seed = parse(Int, option("--seed", string(CapseCambMnuW0WaGeneration.DESIGN_SEED)))
force = "--force" in ARGS

manager = SlurmManager(; launch_timeout=600.0)
addprocs(manager; exeflags=`--startup-file=no`)

@everywhere begin
    using EmulatorsTrainer, PythonCall
    include($GENERATION_FILE)
    using .CapseCambMnuW0WaGeneration
    const CAMB_BACKEND = CapseCambMnuW0WaGeneration.initialize_backend()
    function compute_camb_observables(parameters)
        return CapseCambMnuW0WaGeneration.compute_observables(parameters, CAMB_BACKEND)
    end
end

design = CapseCambMnuW0WaGeneration.create_design(n_samples; seed)
master_backend = CapseCambMnuW0WaGeneration.initialize_backend()
start = time()
dataset_file = compute_dataset_hdf5(
    design, CapseCambMnuW0WaGeneration.PARAMETER_NAMES, output_directory,
    compute_camb_observables;
    mode=:distributed,
    static_arrays=CapseCambMnuW0WaGeneration.static_axes(master_backend), force,
)
metadata = Dict(
    "created_at" => string(now()), "requested_samples" => n_samples,
    "mode" => "Slurm distributed", "workers" => nworkers(),
    "dataset_file" => dataset_file,
    "parameter_names" => CapseCambMnuW0WaGeneration.PARAMETER_NAMES,
    "lower_bounds" => CapseCambMnuW0WaGeneration.LOWER_BOUNDS,
    "upper_bounds" => CapseCambMnuW0WaGeneration.UPPER_BOUNDS,
    "design_seed" => seed, "runtime_seconds" => time() - start,
    "constraint" => "w0 + wa < 0",
    "slurm_job_id" => get(ENV, "SLURM_JOB_ID", nothing),
    "slurm_tasks" => get(ENV, "SLURM_NTASKS", nothing),
    "slurm_nodes" => get(ENV, "SLURM_JOB_NUM_NODES", nothing),
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
