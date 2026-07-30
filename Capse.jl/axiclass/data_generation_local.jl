using ArgParse, Dates, Distributed, EmulatorsTrainer, JSON3, PyCall

s = ArgParseSettings()
@add_arg_table s begin
    "--samples"; arg_type=Int; default=50
    "--output"; default=joinpath(@__DIR__, "data", "smoke_50")
    "--seed"; arg_type=Int; default=20260740
    "--processes"; arg_type=Int; default=2
    "--force"; action=:store_true
end
a = parse_args(s)
n = a["samples"]
output = abspath(a["output"])
n_processes = a["processes"]
n_processes >= 0 || error("--processes must be non-negative")
if n_processes > 0
    project = Base.active_project()
    addprocs(n_processes; exeflags="--project=$project")
end

const GENERATION_FILE = joinpath(@__DIR__, "generation.jl")
include(GENERATION_FILE)
using .AxiclassGeneration
@everywhere using EmulatorsTrainer, PyCall
@everywhere include($GENERATION_FILE)
@everywhere using .AxiclassGeneration
@everywhere compute_axiclass_observables(parameters) = AxiclassGeneration.compute_observables(parameters)

design = AxiclassGeneration.create_design(n; seed=a["seed"])
start = time()
dataset_file = compute_dataset_hdf5(
    design, AxiclassGeneration.PARAMETER_NAMES, output, compute_axiclass_observables;
    mode=(n_processes > 0 ? :distributed : :serial), force=a["force"],
)
metadata = Dict(
    "created_at" => string(now()), "requested_samples" => n,
    "mode" => (n_processes > 0 ? "distributed" : "serial"),
    "processes" => nworkers(), "dataset_file" => dataset_file,
    "parameter_names" => AxiclassGeneration.PARAMETER_NAMES,
    "lower_bounds" => AxiclassGeneration.LOWER_BOUNDS,
    "upper_bounds" => AxiclassGeneration.UPPER_BOUNDS,
    "runtime_seconds" => time() - start,
)
open(joinpath(output, "generation_metadata.json"), "w") do io
    JSON3.write(io, metadata)
end
println("Wrote merged HDF5 dataset: $dataset_file")
