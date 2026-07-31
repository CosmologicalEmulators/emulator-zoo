using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3
using LSFClusterManager

settings = ArgParseSettings()
@add_arg_table settings begin
    "--samples"; arg_type=Int; default=300_000
    "--output"; required=true
    "--seed"; arg_type=Int; default=20260760
    "--workers"; arg_type=Int; default=120
    "--queue"; default="long"
    "--memory-mb"; arg_type=Int; default=4094
    "--force"; action=:store_true
end
arguments = parse_args(settings)
project = Base.active_project()
flags = `-q $(arguments["queue"]) -n 1 -M $(arguments["memory-mb"])`
addprocs_lsf(arguments["workers"]; bsub_flags=flags, exeflags="--project=$project")

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
@everywhere include($GENERATION_FILE)
@everywhere using .ACEClassMnuW0WaGeneration

design = ACEClassMnuW0WaGeneration.create_design(arguments["samples"]; seed=arguments["seed"])
retained_samples = size(design, 2)
println("Retained $retained_samples of $(arguments["samples"]) LHS candidates after w0 + wa <= 0 rejection")
output_directory = abspath(arguments["output"])
start = time()
dataset_file = compute_dataset_hdf5(
    design,
    ACEClassMnuW0WaGeneration.PARAMETER_NAMES,
    output_directory,
    ACEClassMnuW0WaGeneration.compute_observables;
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
        "constraint" => "w0 + wa <= 0",
        "failure_log" => joinpath(output_directory, "generation_failures.json"),
        "workers" => arguments["workers"],
        "queue" => arguments["queue"],
        "memory_mb" => arguments["memory-mb"],
        "seed" => arguments["seed"],
        "dataset_file" => dataset_file,
        "runtime_seconds" => time() - start,
    ))
end
println("Wrote merged HDF5 dataset: $dataset_file")
