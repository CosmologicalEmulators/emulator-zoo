using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3
using LSFClusterManager

settings = ArgParseSettings()
@add_arg_table settings begin
    "--samples"; arg_type=Int; default=250_000
    "--output"; required=true
    "--seed"; arg_type=Int; default=20260763
    "--workers"; arg_type=Int; default=90
    "--queue"; default="long"
    "--memory-mb"; arg_type=Int; default=8192
    "--force"; action=:store_true
end
arguments = parse_args(settings)
project = Base.active_project()
flags = `-q $(arguments["queue"]) -n 1 -M $(arguments["memory-mb"])`
addprocs_lsf(arguments["workers"]; bsub_flags=flags, exeflags="--project=$project")

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
@everywhere include($GENERATION_FILE)
@everywhere using .MapseClassMnuW0WaGeneration

design = MapseClassMnuW0WaGeneration.create_design(arguments["samples"]; seed=arguments["seed"])
output_directory = abspath(arguments["output"])
start = time()
dataset_file = compute_dataset_hdf5(
    design,
    MapseClassMnuW0WaGeneration.PARAMETER_NAMES,
    output_directory,
    MapseClassMnuW0WaGeneration.compute_observables;
    mode=:distributed,
    force=arguments["force"],
    static_arrays=(k=MapseClassMnuW0WaGeneration.K_GRID,),
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, Dict(
        "created_at" => string(now()),
        "requested_samples" => arguments["samples"],
        "workers" => arguments["workers"],
        "queue" => arguments["queue"],
        "memory_mb" => arguments["memory-mb"],
        "seed" => arguments["seed"],
        "dataset_file" => dataset_file,
        "runtime_seconds" => time() - start,
    ))
end
println("Wrote merged HDF5 dataset: $dataset_file")
