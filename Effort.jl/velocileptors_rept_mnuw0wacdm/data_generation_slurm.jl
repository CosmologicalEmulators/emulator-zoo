using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3
using SlurmClusterManager

settings = ArgParseSettings()
@add_arg_table settings begin
    "--samples"; arg_type=Int; default=200_000
    "--output"; required=true
    "--seed"; arg_type=Int; default=20260744
    "--force"; action=:store_true
end
arguments = parse_args(settings)

manager = SlurmManager(; launch_timeout=300.0)
addprocs(
    manager;
    # SlurmClusterManager propagates JULIA_PROJECT to workers. Do not pass a
    # space-separated --project string as one exeflags argument.
    exeflags=`--startup-file=no`,
)

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
@everywhere include($GENERATION_FILE)
@everywhere using .VelocileptorsREPTMnuW0WaGeneration
@everywhere const VELOCILEPTORS_BACKEND = initialize_backend()
@everywhere function compute_velocileptors_observables(parameters)
    return compute_observables(parameters, VELOCILEPTORS_BACKEND)
end

design = create_design(arguments["samples"]; seed=arguments["seed"])
retained_samples = size(design, 2)
println(
    "Retained $retained_samples of $(arguments["samples"]) LHS candidates " *
    "after w0 + wa < 0 rejection",
)
output_directory = abspath(arguments["output"])
start = time()
dataset_file = EmulatorsTrainer.compute_dataset_hdf5(
    design,
    PARAMETER_NAMES,
    output_directory,
    compute_velocileptors_observables;
    mode=:distributed,
    force=arguments["force"],
    skip_errors=true,
)
dataset = load_hdf5_dataset(dataset_file)
successful_samples = size(dataset.parameters, 1)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, Dict(
        "created_at" => string(now()),
        "candidate_samples" => arguments["samples"],
        "retained_samples" => retained_samples,
        "successful_samples" => successful_samples,
        "failed_samples" => retained_samples - successful_samples,
        "acceptance_fraction" => retained_samples / arguments["samples"],
        "constraint" => "w0 + wa < 0",
        "failure_log" => joinpath(output_directory, "generation_failures.json"),
        "workers" => nworkers(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", nothing),
        "slurm_tasks" => get(ENV, "SLURM_NTASKS", nothing),
        "slurm_nodes" => get(ENV, "SLURM_JOB_NUM_NODES", nothing),
        "seed" => arguments["seed"],
        "dataset_file" => dataset_file,
        "runtime_seconds" => time() - start,
    ))
end
println("Wrote merged HDF5 dataset: $dataset_file")
