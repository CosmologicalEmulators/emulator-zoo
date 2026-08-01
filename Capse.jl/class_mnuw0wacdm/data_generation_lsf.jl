using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3
using LSFClusterManager

include(joinpath(@__DIR__, "generation.jl"))
using .ClassMnuW0WaGeneration

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--samples"; arg_type=Int; default=200_000
        "--output"; required=true
        "--workers"; arg_type=Int; default=80
        "--queue"; default="long"
        "--memory-mb"; arg_type=Int; default=4096
        "--seed"; arg_type=Int; default=20260752
        "--force"; action=:store_true
    end
    return parse_args(settings)
end

function main()
    arguments = parse_commandline()
    project_directory = @__DIR__
    generation_file = joinpath(@__DIR__, "generation.jl")
    workers = addprocs_lsf(
        arguments["workers"];
        bsub_flags=`-q $(arguments["queue"]) -n 1 -M $(arguments["memory-mb"])`,
        exeflags="--project=$project_directory",
    )
    try
        @everywhere include($generation_file)
        @everywhere using .ClassMnuW0WaGeneration
        @everywhere const CLASS_BACKEND = initialize_backend()
        @everywhere function compute_class_observables(parameters)
            return compute_observables(parameters, CLASS_BACKEND)
        end
        design = create_design(arguments["samples"]; seed=arguments["seed"])
        output_directory = abspath(arguments["output"])
        start = time()
        dataset_file = compute_dataset_hdf5(
            design, PARAMETER_NAMES, output_directory, compute_class_observables;
            mode=:distributed, force=arguments["force"], skip_errors=true,
        )
        dataset = load_hdf5_dataset(dataset_file)
        open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
            JSON3.write(stream, Dict(
                "created_at" => string(now()),
                "requested_samples" => arguments["samples"],
                "successful_samples" => size(dataset.parameters, 1),
                "failed_samples" => arguments["samples"] - size(dataset.parameters, 1),
                "mode" => "LSF distributed",
                "workers" => arguments["workers"], "queue" => arguments["queue"],
                "dataset_file" => dataset_file, "seed" => arguments["seed"],
                "runtime_seconds" => time() - start,
            ))
        end
        println("Wrote merged HDF5 dataset: $dataset_file")
    finally
        rmprocs(workers)
    end
end

main()
