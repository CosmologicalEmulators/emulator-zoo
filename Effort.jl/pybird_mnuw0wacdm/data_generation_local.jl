using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3

settings = ArgParseSettings()
@add_arg_table settings begin
    "--samples"; arg_type=Int; default=50
    "--output"; default=joinpath(@__DIR__, "data", "smoke_50")
    "--seed"; arg_type=Int; default=20260763
    "--processes"; arg_type=Int; default=2
    "--force"; action=:store_true
end
arguments = parse_args(settings)
n_processes = arguments["processes"]
n_processes >= 0 || error("--processes must be non-negative")
if n_processes > 0
    addprocs(n_processes; exeflags="--project=$(Base.active_project())")
end

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
@everywhere include($GENERATION_FILE)
@everywhere using .PyBirdMnuW0WaGeneration

design = PyBirdMnuW0WaGeneration.create_design(arguments["samples"]; seed=arguments["seed"])
output_directory = abspath(arguments["output"])
start = time()
dataset_file = compute_dataset_hdf5(
    design, PARAMETER_NAMES, output_directory, compute_observables;
    mode=(n_processes > 0 ? :distributed : :serial),
    force=arguments["force"], static_arrays=(kk=K_GRID, kd=KD_GRID), skip_errors=true,
)
dataset = load_hdf5_dataset(dataset_file)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, Dict(
        "created_at" => string(now()), "requested_samples" => arguments["samples"],
        "successful_samples" => size(dataset.parameters, 1),
        "failed_samples" => arguments["samples"] - size(dataset.parameters, 1),
        "processes" => nworkers(), "seed" => arguments["seed"],
        "dataset_file" => dataset_file, "runtime_seconds" => time() - start,
    ))
end
println("Wrote merged HDF5 dataset: $dataset_file")
