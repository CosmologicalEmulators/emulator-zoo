using AbstractCosmologicalEmulators, ArgParse, DataFrames, Effort, EmulatorsTrainer, JSON3, NPZ

const INPUT_COLUMNS = [:z, :ln10As, :ns, :H0, :ombh2, :omch2, :Mν, :ωk]

function parse_commandline()
    s=ArgParseSettings()
    @add_arg_table s begin
        "--component"; default="loop"
        "--multipole", "-l"; arg_type=Int; default=0
        "--path-input", "-i"; required=true
        "--path-output", "-o"; required=true
        "--steps-per-session"; arg_type=Int; default=2000
        "--sessions-per-rate"; arg_type=Int; default=10
        "--batch-size"; arg_type=Int; default=256
        "--split-seed"; arg_type=Int; default=20260742
    end
    parse_args(s)
end

component_columns(c) = c == "11" ? (1:3) : c == "loop" ? (4:12) : c == "ct" ? (13:16) : error("Bad component $c")

function factor(p, component)
    h=p["H0"]/100; ωk=p["Omega_k"]*h^2
    cosmo=Effort.w0waCDMCosmology(ln10Aₛ=3.0,nₛ=0.96,h=h,ωb=p["ombh2"],ωc=p["omch2"],mν=p["Mν"],ωk=ωk)
    base=exp(p["ln10As"])*1e-10*Effort.D_z(p["z"],cosmo)^2
    component=="loop" ? base^2 : base
end

function copy_template(source,dest,out)
    isfile(joinpath(@__DIR__,source)) || error("Missing template $source")
    cp(joinpath(@__DIR__,source),joinpath(out,dest);force=true)
end

function main()
    a=parse_commandline(); c=a["component"]; ell=a["multipole"]; cols=component_columns(c)
    input=abspath(a["path-input"]); out=joinpath(abspath(a["path-output"]),string(ell),c); mkpath(out)
    dataset=load_hdf5_dataset(input); all(dataset.valid)||error("HDF5 dataset contains invalid samples")
    pa=dataset.parameters; pn=dataset.parameter_names; obs=get(dataset.observables,Symbol("pk_$(ell)"),nothing)
    obs===nothing && error("Observable pk_$ell is not present in $input")
    df=DataFrame(sample_id=String[],z=Float64[],ln10As=Float64[],ns=Float64[],H0=Float64[],ombh2=Float64[],omch2=Float64[],Mν=Float64[],ωk=Float64[],observable=Vector{Float64}[])
    for i in axes(pa,1)
        p=Dict(pn[j]=>pa[i,j] for j in axes(pa,2)); h=p["H0"]/100
        push!(df,(sample_id="sample_$(lpad(i,6,'0'))",z=Float64(p["z"]),ln10As=Float64(p["ln10As"]),ns=Float64(p["ns"]),H0=Float64(p["H0"]),ombh2=Float64(p["ombh2"]),omch2=Float64(p["omch2"]),Mν=Float64(p["Mν"]),ωk=Float64(p["Omega_k"]*h^2),observable=vec(Array(obs[i,:,:])[:,cols])./factor(p,c)))
    end
    report=(loaded=size(df,1),skipped=0); report.loaded>=2 || error("Too few samples")
    inmm=get_minmax_in(df,INPUT_COLUMNS); _,y=extract_input_output_df(df;input_columns=INPUT_COLUMNS); outmm=get_minmax_out(y)
    maximin_df!(df,inmm,outmm;input_columns=INPUT_COLUMNS)
    xt,yt,xv,yv,ti,vi=getdata(df;seed=a["split-seed"],input_columns=INPUT_COLUMNS,return_indices=true)
    nd=size(yt,1); nn=Dict{String,Any}("n_input_features"=>8,"n_output_features"=>nd,"n_hidden_layers"=>5,"layers"=>Dict("layer_$i"=>Dict("n_neurons"=>64,"activation_function"=>"tanh") for i=1:5),"emulator_description"=>Dict("source"=>"CLASS + Velocileptors LPT","component"=>c,"multipole"=>ell))
    network=AbstractCosmologicalEmulators._get_nn_simplechains(nn)
    npzwrite(joinpath(out,"inminmax.npy"),inmm);npzwrite(joinpath(out,"outminmax.npy"),outmm)
    npzwrite(joinpath(out,"k.npy"),vec(dataset.observables[:kv][1,:]));npzwrite(joinpath(out,"train_indices.npy"),ti.-1);npzwrite(joinpath(out,"validation_indices.npy"),vi.-1)
    open(joinpath(out,"nn_setup.json"),"w") do io;JSON3.write(io,nn);end
    copy_template(c=="loop" ? "postprocessing_loop.jl" : "postprocessing.jl","postprocessing.jl",out)
    copy_template(c=="loop" ? "postprocessing_loop.py" : "postprocessing.py","postprocessing.py",out)
    copy_template("stochmodel_$(ell).jl","stochmodel.jl",out);copy_template("stochmodel_$(ell).py","stochmodel.py",out)
    config=SimpleChainsTrainingConfig(learning_rates=[1e-4,7e-5,5e-5,2e-5,1e-5,7e-6,5e-6,2e-6,1e-6,7e-7],sessions_per_rate=a["sessions-per-rate"],steps_per_session=a["steps-per-session"],batch_size=a["batch-size"],initialization_seed=20260743)
    cb=p->(println("steps=$(p.total_steps) validation=$(p.validation_loss)");flush(stdout))
    result=train_simplechains(network,xt,yt,xv,yv;config,callback=cb,
        checkpoint_callback=(parameters, progress) ->
            save_training_checkpoint(out, parameters, progress))
    save_training_result(out,result;metadata=Dict("n_loaded"=>report.loaded,"n_train"=>length(ti),"n_validation"=>length(vi),"component"=>c,"multipole"=>ell))
end
main()
