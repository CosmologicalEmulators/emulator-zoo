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
workers = addprocs_lsf(arguments["workers"]; bsub_flags=`-q $(arguments["queue"]) -n 1 -M $(arguments["memory-mb"])`, exeflags="--project=$project")
try
    const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
    @everywhere include($GENERATION_FILE)
    @everywhere using .PyBirdMnuW0WaGeneration
    design = create_design(arguments["samples"]; seed=arguments["seed"])
    output_directory = abspath(arguments["output"])
    start = time()
    dataset_file = compute_dataset_hdf5(
        design, PARAMETER_NAMES, output_directory, compute_observables;
        mode=:distributed, force=arguments["force"],
        static_arrays=(kk=K_GRID, kd=KD_GRID), skip_errors=true,
    )
    dataset = load_hdf5_dataset(dataset_file)
    open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
        JSON3.write(stream, Dict(
            "created_at" => string(now()), "requested_samples" => arguments["samples"],
            "successful_samples" => size(dataset.parameters, 1),
            "failed_samples" => arguments["samples"] - size(dataset.parameters, 1),
            "workers" => arguments["workers"], "queue" => arguments["queue"],
            "memory_mb" => arguments["memory-mb"], "seed" => arguments["seed"],
            "dataset_file" => dataset_file, "runtime_seconds" => time() - start,
        ))
    end
    println("Wrote merged HDF5 dataset: $dataset_file")
finally
    rmprocs(workers)
end
