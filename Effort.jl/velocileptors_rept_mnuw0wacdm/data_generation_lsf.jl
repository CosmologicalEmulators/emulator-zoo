using ArgParse, Distributed, EmulatorsTrainer, JSON3, LSFClusterManager, NPZ
include(joinpath(@__DIR__,"generation.jl"));using .VelocileptorsREPTMnuW0WaGeneration
s=ArgParseSettings()
@add_arg_table s begin
    "--samples";arg_type=Int;default=200000
    "--output";required=true
    "--workers";arg_type=Int;default=80
    "--queue";default="long"
    "--memory-mb";arg_type=Int;default=4096
    "--seed";arg_type=Int;default=20260744
    "--force";action=:store_true
end
a=parse_args(s)
workers=addprocs_lsf(a["workers"];bsub_flags=`-q $(a["queue"]) -n 1 -M $(a["memory-mb"])`,exeflags="--project=$(@__DIR__)")
try
    @everywhere include($(joinpath(@__DIR__,"generation.jl")))
    @everywhere using .VelocileptorsREPTMnuW0WaGeneration
    @everywhere const BACKEND=initialize_backend()
    @everywhere function generate(p,r)
        try;write_sample(r,p,compute_observables(p,BACKEND));true
        catch e;@warn "Skipping failed sample" exception=(e,catch_backtrace());false end
    end
    d=create_design(a["samples"];seed=a["seed"]);out=abspath(a["output"])
    compute_dataset(d,PARAMETER_NAMES,out,generate,:distributed;force=a["force"])
    npzwrite(joinpath(out,"design.npy"),d)
finally
    rmprocs(workers)
end
