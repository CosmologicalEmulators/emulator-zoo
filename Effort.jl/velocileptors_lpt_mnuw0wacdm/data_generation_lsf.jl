using ArgParse
using Distributed
using EmulatorsTrainer
using JSON3
using LSFClusterManager
using NPZ

include(joinpath(@__DIR__, "generation.jl"))
using .VelocileptorsLPTMnuW0WaGeneration

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
        default = 20260738
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
        @everywhere using .VelocileptorsLPTMnuW0WaGeneration
        @everywhere const VELOCILEPTORS_BACKEND = initialize_backend()
        @everywhere function generate_velocileptors_sample(parameters, root)
            try
                observables = compute_observables(parameters, VELOCILEPTORS_BACKEND)
                write_sample(root, parameters, observables)
                return true
            catch error
                @warn "Skipping failed Velocileptors sample" exception=(error, catch_backtrace())
                return false
            end
        end

        design = create_design(arguments["samples"]; seed=arguments["seed"])
        compute_dataset(
            design,
            PARAMETER_NAMES,
            output_directory,
            generate_velocileptors_sample,
            :distributed;
            force=arguments["force"],
        )
        npzwrite(joinpath(output_directory, "design.npy"), design)
        successful = count(
            name -> startswith(name, "sample_") && isdir(joinpath(output_directory, name)),
            readdir(output_directory),
        )
        open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
            JSON3.write(stream, Dict(
                "requested_samples" => arguments["samples"],
                "successful_samples" => successful,
                "failed_samples" => arguments["samples"] - successful,
                "parameter_names" => PARAMETER_NAMES,
                "constraint" => "w0 + wa < 0",
                "seed" => arguments["seed"],
                "execution" => "LSF distributed",
                "workers" => arguments["workers"],
                "queue" => arguments["queue"],
            ))
        end
    finally
        rmprocs(worker_ids)
    end
end

main()
