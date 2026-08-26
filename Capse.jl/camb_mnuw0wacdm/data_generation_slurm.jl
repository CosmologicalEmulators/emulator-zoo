using Dates, Distributed, EmulatorsTrainer, HDF5, JSON3, PyCall, SlurmClusterManager, Statistics

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
output_argument = option("--output", nothing)
output_argument === nothing && error("--output is required")
output_directory = abspath(output_argument)
seed = parse(Int, option("--seed", string(CapseCambMnuW0WaGeneration.DESIGN_SEED)))
force = "--force" in ARGS

master_backend = CapseCambMnuW0WaGeneration.initialize_backend()
configuration = CapseCambMnuW0WaGeneration.backend_configuration(master_backend)
println("PyCall Python: ", PyCall.python)
println("CAMB path: ", configuration["camb_path"])
println("CAMB version: ", configuration["camb_version"])
println("Recombination model: ", configuration["recombination_model"])

manager = SlurmManager(; launch_timeout=600.0)
project = Base.active_project()
addprocs(manager; exeflags=`--startup-file=no --project=$project`)

@everywhere begin
    using EmulatorsTrainer, PyCall
    include($GENERATION_FILE)
    using .CapseCambMnuW0WaGeneration
    const CAMB_BACKEND = Ref{Any}(nothing)
    function compute_camb_observables(parameters)
        backend = CAMB_BACKEND[]
        if backend === nothing
            backend = CapseCambMnuW0WaGeneration.initialize_backend()
            CAMB_BACKEND[] = backend
        end
        return CapseCambMnuW0WaGeneration.compute_observables(parameters, backend)
    end
end

design = CapseCambMnuW0WaGeneration.create_design(n_samples; seed)
start = time()
dataset_file = compute_dataset_hdf5(
    design, CapseCambMnuW0WaGeneration.PARAMETER_NAMES, output_directory,
    compute_camb_observables;
    mode=:distributed,
    static_arrays=CapseCambMnuW0WaGeneration.static_axes(master_backend),
    force, skip_errors=true,
)
retained_samples = h5open(dataset_file, "r") do file
    length(file["sample_indices"])
end
failure_file = joinpath(output_directory, "generation_failures.json")
failures = isfile(failure_file) ? JSON3.read(read(failure_file, String)) : []
w0 = view(design, 8, :)
wa = view(design, 9, :)
metadata = Dict(
    "created_at" => string(now()), "requested_samples" => n_samples,
    "mode" => "Slurm distributed", "workers" => nworkers(),
    "dataset_file" => dataset_file,
    "parameter_names" => CapseCambMnuW0WaGeneration.PARAMETER_NAMES,
    "lower_bounds" => CapseCambMnuW0WaGeneration.LOWER_BOUNDS,
    "upper_bounds" => CapseCambMnuW0WaGeneration.UPPER_BOUNDS,
    "design_seed" => seed,
    "retained_samples" => retained_samples,
    "failed_samples" => length(failures),
    "w0_wa_correlation" => length(w0) > 1 ? cor(w0, wa) : nothing,
    "w0_plus_wa_min" => minimum(w0 .+ wa),
    "w0_plus_wa_max" => maximum(w0 .+ wa),
    "python_executable" => PyCall.python,
    "camb_configuration" => configuration,
    "runtime_seconds" => time() - start,
    "constraint" => "w0 + wa < -0.5",
    "slurm_job_id" => get(ENV, "SLURM_JOB_ID", nothing),
    "slurm_tasks" => get(ENV, "SLURM_NTASKS", nothing),
    "slurm_nodes" => get(ENV, "SLURM_JOB_NUM_NODES", nothing),
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
