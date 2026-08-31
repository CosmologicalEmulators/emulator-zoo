using ArgParse
using Distributed
using EmulatorsTrainer
using HDF5
using JSON3

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .FolpsMnuW0WaGeneration

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "samples"; arg_type=Int
        "output"
        "--processes"; arg_type=Int; default=2
        "--seed"; arg_type=Int; default=FolpsMnuW0WaGeneration.DESIGN_SEED
        "--force"; action=:store_true
    end
    return parse_args(settings)
end

arguments = parse_commandline()
n_processes = arguments["processes"]
n_processes >= 0 || error("--processes must be non-negative")
project = Base.active_project()
n_processes > 0 && addprocs(n_processes; exeflags=`--startup-file=no --project=$project`)

@everywhere begin
    using PyCall
    include($GENERATION_FILE)
    using .FolpsMnuW0WaGeneration
    const FOLPS_BACKEND = Ref{Any}(nothing)
    function compute_folps_observables(parameters)
        backend = FOLPS_BACKEND[]
        if backend === nothing
            backend = FolpsMnuW0WaGeneration.initialize_backend()
            FOLPS_BACKEND[] = backend
        end
        return FolpsMnuW0WaGeneration.compute_observables(parameters, backend)
    end
end

design = FolpsMnuW0WaGeneration.create_design(arguments["samples"]; seed=arguments["seed"])
master_backend = FolpsMnuW0WaGeneration.initialize_backend()
output = abspath(arguments["output"])
dataset = compute_dataset_hdf5(
    design,
    FolpsMnuW0WaGeneration.PARAMETER_NAMES,
    output,
    compute_folps_observables;
    mode=n_processes > 0 ? :distributed : :serial,
    static_arrays=FolpsMnuW0WaGeneration.static_axes(master_backend),
    force=arguments["force"],
    skip_errors=true,
)

n_retained = h5open(dataset, "r") do file
    size(file["parameters"], 1)
end
failures_path = joinpath(output, "generation_failures.json")
failures = isfile(failures_path) ? JSON3.read(read(failures_path, String)) : []
metadata = Dict(
    "requested_samples" => arguments["samples"],
    "retained_samples" => n_retained,
    "failed_samples" => length(failures),
    "processes" => nworkers(),
    "parameter_names" => FolpsMnuW0WaGeneration.PARAMETER_NAMES,
    "lower_bounds" => FolpsMnuW0WaGeneration.LOWER_BOUNDS,
    "upper_bounds" => FolpsMnuW0WaGeneration.UPPER_BOUNDS,
    "design_seed" => arguments["seed"],
    "table_layout" => "AP-free IR-resummed multipole bias basis",
)
open(joinpath(output, "generation_metadata.json"), "w") do stream
    JSON3.write(stream, metadata)
end
println("Wrote Folps EFT dataset: $dataset")
