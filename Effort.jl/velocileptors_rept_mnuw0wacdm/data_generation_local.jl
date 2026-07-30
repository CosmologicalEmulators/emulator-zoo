using ArgParse, Dates, Distributed, EmulatorsTrainer, JSON3

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--samples"; arg_type=Int; default=50
        "--output"; default=joinpath(@__DIR__, "data", "smoke_50")
        "--seed"; arg_type=Int; default=20260744
        "--processes"; arg_type=Int; default=2
        "--force"; action=:store_true
    end
    return parse_args(settings)
end

arguments = parse_commandline()
n_samples = arguments["samples"]
output_directory = abspath(arguments["output"])
n_processes = arguments["processes"]
n_processes >= 0 || error("--processes must be non-negative")

if n_processes > 0
    project = Base.active_project()
    addprocs(n_processes; exeflags="--project=$project")
end

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .VelocileptorsREPTMnuW0WaGeneration
@everywhere using EmulatorsTrainer, PyCall, Random
@everywhere include($GENERATION_FILE)
@everywhere using .VelocileptorsREPTMnuW0WaGeneration
@everywhere function compute_velocileptors_observables(parameters)
    backend = VelocileptorsREPTMnuW0WaGeneration.initialize_backend()
    return VelocileptorsREPTMnuW0WaGeneration.compute_observables(parameters, backend)
end

design = VelocileptorsREPTMnuW0WaGeneration.create_design(n_samples; seed=arguments["seed"])
start = time()
dataset_file = EmulatorsTrainer.compute_dataset_hdf5(
    design,
    VelocileptorsREPTMnuW0WaGeneration.PARAMETER_NAMES,
    output_directory,
    compute_velocileptors_observables;
    mode=(n_processes > 0 ? :distributed : :serial),
    force=arguments["force"],
)
metadata = Dict(
    "created_at" => string(now()),
    "requested_samples" => n_samples,
    "mode" => (n_processes > 0 ? "distributed" : "serial"),
    "processes" => nworkers(),
    "dataset_file" => dataset_file,
    "parameter_names" => VelocileptorsREPTMnuW0WaGeneration.PARAMETER_NAMES,
    "lower_bounds" => VelocileptorsREPTMnuW0WaGeneration.LOWER_BOUNDS,
    "upper_bounds" => VelocileptorsREPTMnuW0WaGeneration.UPPER_BOUNDS,
    "seed" => arguments["seed"],
    "runtime_seconds" => time() - start,
)
open(joinpath(output_directory, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
