using ArgParse
using Dates
using Distributed
using EmulatorsTrainer
using JSON3
using LSFClusterManager

include(joinpath(@__DIR__, "generation.jl"))
using .VelocileptorsREPTMnuW0WaGeneration

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--samples"
        arg_type = Int
        default = 200_000
        "--output"
        required = true
        "--workers"
        arg_type = Int
        default = 80
        "--queue"
        default = "long"
        "--memory-mb"
        arg_type = Int
        default = 4_096
        "--seed"
        arg_type = Int
        default = 20260744
        "--force"
        action = :store_true
    end
    return parse_args(settings)
end

function main()
    arguments = parse_commandline()
    output_directory = abspath(arguments["output"])
    project_directory = @__DIR__
    generation_file = joinpath(@__DIR__, "generation.jl")
    flags = `-q $(arguments["queue"]) -n 1 -M $(arguments["memory-mb"])`
    worker_ids = addprocs_lsf(
        arguments["workers"];
        bsub_flags=flags,
        exeflags="--project=$project_directory",
    )
    try
        @everywhere include($generation_file)
        @everywhere using .VelocileptorsREPTMnuW0WaGeneration
        @everywhere const VELOCILEPTORS_BACKEND = initialize_backend()
        @everywhere function compute_velocileptors_observables(parameters)
            return compute_observables(parameters, VELOCILEPTORS_BACKEND)
        end

        design = create_design(arguments["samples"]; seed=arguments["seed"])
        start = time()
        dataset_file = EmulatorsTrainer.compute_dataset_hdf5(
            design,
            PARAMETER_NAMES,
            output_directory,
            compute_velocileptors_observables;
            mode=:distributed,
            force=arguments["force"],
        )
        open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
            JSON3.write(stream, Dict(
                "created_at" => string(now()),
                "requested_samples" => arguments["samples"],
                "mode" => "LSF distributed",
                "workers" => arguments["workers"],
                "queue" => arguments["queue"],
                "dataset_file" => dataset_file,
                "parameter_names" => PARAMETER_NAMES,
                "lower_bounds" => LOWER_BOUNDS,
                "upper_bounds" => UPPER_BOUNDS,
                "seed" => arguments["seed"],
                "runtime_seconds" => time() - start,
            ))
        end
        println("Wrote merged HDF5 dataset: $dataset_file")
    finally
        rmprocs(worker_ids)
    end
end

main()
