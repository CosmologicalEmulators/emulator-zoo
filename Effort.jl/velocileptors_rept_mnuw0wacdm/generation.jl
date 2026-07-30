module VelocileptorsREPTMnuW0WaGeneration
using EmulatorsTrainer, JSON3, NPZ, PyCall, Random, SHA
export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, create_design, initialize_backend, compute_observables, write_sample
const PARAMETER_NAMES=["z","ln10As","ns","H0","ombh2","omch2","Mν","w0","wa"]
const LOWER_BOUNDS=[0.285,2.0,0.8,50.0,0.02,0.08,0.0,-3.0,-3.0]
const UPPER_BOUNDS=[1.9,3.5,1.10,90.0,0.025,0.18,0.5,0.5,2.0]
struct Backend; classy::PyObject; rept::PyObject; pnw::PyObject; konh::Vector{Float64}; kv::Vector{Float64}; end
function create_design(n;seed=20260744)
    Random.seed!(seed); d=create_training_dataset(n,LOWER_BOUNDS,UPPER_BOUNDS); w0=view(d,8,:); wa=copy(view(d,9,:))
    for (i,j) in zip(sortperm(w0),sortperm(wa;rev=true)); d[9,i]=wa[j]; end
    all(d[8,:].+d[9,:].<0)||error("w0 + wa constraint failed"); d
end
function initialize_backend()
    konh=10.0.^range(-3,1;length=20_000); kv=10.0.^range(log10(5e-4),log10(.5);length=80)
    Backend(pyimport("classy"),pyimport("velocileptors.EPT.ept_fullresum_fftw"),pyimport("velocileptors.Utils.pnw_dst"),konh,kv)
end
function compute_observables(p,b::Backend)
    h=p["H0"]/100; c=b.classy.Class()
    try
        c.set(Dict("output"=>"mPk","P_k_max_h/Mpc"=>20.0,"z_pk"=>"0.0,3.0","h"=>h,"omega_b"=>p["ombh2"],"omega_cdm"=>p["omch2"],"ln10^{10}A_s"=>p["ln10As"],"n_s"=>p["ns"],"tau_reio"=>.0568,"N_ur"=>2.033,"N_ncdm"=>1,"m_ncdm"=>p["Mν"],"use_ppf"=>"yes","w0_fld"=>p["w0"],"wa_fld"=>p["wa"],"fluid_equation_of_state"=>"CLP","cs2_fld"=>1.0,"Omega_Lambda"=>0.0)); c.compute()
        f=Float64(c.scale_independent_growth_factor_f(p["z"])); plin=[Float64(c.pk_cb(k*h,p["z"])) * h^3 for k in b.konh]; knw,pnw=b.pnw.pnw_dst(b.konh,plin)
        m=b.rept.REPT(knw,plin;pnw=pnw,kmin=5e-4,kmax=.5,nk=80,beyond_gauss=true,one_loop=true,N=2000,extrap_min=-6,extrap_max=2,cutoff=100,threads=1)
        m.compute_redshift_space_power_multipoles_tables(f;apar=1.0,aperp=1.0,ngauss=4)
        r=(kv=Vector{Float64}(m.kv),pk_lin=plin,pk_0=Array(m.p0ktable),pk_2=Array(m.p2ktable),pk_4=Array(m.p4ktable),knw=knw,Pnw=pnw)
        all(x->all(isfinite,x),r)||error("REPT output contains NaN or Inf"); r
    finally; try c.struct_cleanup();c.empty() catch end; end
end
function write_sample(root,p,r)
    text=join(("$k=$(p[k])" for k in PARAMETER_NAMES),";"); dir=joinpath(root,"sample_"*bytes2hex(sha1(text))[1:16]); mkdir(dir)
    for k in (:kv,:pk_lin,:pk_0,:pk_2,:pk_4,:knw,:Pnw); npzwrite(joinpath(dir,"$k.npy"),getproperty(r,k)); end
    q=Dict{String,Any}(p);q["sample_id"]=basename(dir);open(joinpath(dir,"effort_dict.json"),"w") do io;JSON3.write(io,q);end;dir
end
end
