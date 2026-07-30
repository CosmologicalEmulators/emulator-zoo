using ArgParse, Distributed, EmulatorsTrainer, JSON3, LSFClusterManager, NPZ
include(joinpath(@__DIR__, "generation.jl")); using .VelocileptorsLPTMnuOmegaKGeneration
s=ArgParseSettings(); @add_arg_table s begin
    "--samples"; arg_type=Int; default=200000
    "--output"; required=true
    "--workers"; arg_type=Int; default=80
    "--queue"; default="long"
    "--memory-mb"; arg_type=Int; default=4096
    "--seed"; arg_type=Int; default=20260741
    "--force"; action=:store_true
end
a=parse_args(s); project=@__DIR__; workers=addprocs_lsf(a["workers"];bsub_flags=`-q $(a["queue"]) -n 1 -M $(a["memory-mb"])`,exeflags="--project=$project")
try
    @everywhere include($(joinpath(@__DIR__,"generation.jl"))); @everywhere using .VelocileptorsLPTMnuOmegaKGeneration
    @everywhere const BACKEND=initialize_backend()
    @everywhere function generate(p,root)
        try; write_sample(root,p,compute_observables(p,BACKEND)); true
        catch e; @warn "Skipping failed sample" exception=(e,catch_backtrace()); false end
    end
    design=create_design(a["samples"];seed=a["seed"]); output=abspath(a["output"])
    compute_dataset(design,PARAMETER_NAMES,output,generate,:distributed;force=a["force"])
    npzwrite(joinpath(output,"design.npy"),design)
    open(joinpath(output,"generation_metadata.json"),"w") do io
        JSON3.write(io,Dict("requested_samples"=>a["samples"],"execution"=>"LSF distributed","workers"=>a["workers"],"seed"=>a["seed"]))
    end
finally
    rmprocs(workers)
end
