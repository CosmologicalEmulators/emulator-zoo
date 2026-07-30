module VelocileptorsREPTCDMGeneration
using EmulatorsTrainer, JSON3, NPZ, PyCall, Random, SHA
export PARAMETER_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, create_design, initialize_backend, compute_observables, write_sample
const PARAMETER_NAMES=["ln10As","H0","omch2"]
const LOWER_BOUNDS=[2.6,60.0,0.08]
const UPPER_BOUNDS=[3.4,74.0,0.16]
struct Backend; classy::PyObject; rept::PyObject; pnw::PyObject; special::PyObject; konh::Vector{Float64}; kv::Vector{Float64}; end
create_design(n;seed=20260748)=(Random.seed!(seed);create_training_dataset(n,LOWER_BOUNDS,UPPER_BOUNDS))
function initialize_backend()
    konh=10.0.^range(-3,1;length=20_000); kv=vcat(5e-4,10.0.^range(log10(1.5e-3),log10(.025);length=10),collect(.03:.01:.5))
    Backend(pyimport("classy"),pyimport("velocileptors.EPT.ept_fullresum_fftw"),pyimport("velocileptors.Utils.pnw_dst"),pyimport("scipy.special"),konh,kv)
end
function growth_D(a,OmegaM,special)
    Float64(a*special.hyp2f1(1/3,1,11/6,-a^3/OmegaM*(1-OmegaM))/special.hyp2f1(1/3,1,11/6,-1/OmegaM*(1-OmegaM)))
end
function growth_f(a,OmegaM,special)
    da=growth_D(a,OmegaM,special)
    ret=da/a - a*(6*a^2*(1-OmegaM)*Float64(special.hyp2f1(4/3,2,17/6,-a^3*(1-OmegaM)/OmegaM)))/(11*OmegaM)/Float64(special.hyp2f1(1/3,1,11/6,-1/OmegaM*(1-OmegaM)))
    a*ret/da
end
function class_params(p)
    Dict("output"=>"mPk","P_k_max_h/Mpc"=>20.0,"z_pk"=>"0.0,10","h"=>p["H0"]/100,"omega_b"=>0.02237,"omega_cdm"=>p["omch2"],"ln10^{10}A_s"=>p["ln10As"],"n_s"=>0.9649,"tau_reio"=>0.0568,"N_ur"=>2.033,"N_ncdm"=>1,"m_ncdm"=>0.06)
end
function fid_params()
    Dict("output"=>"mPk","P_k_max_h/Mpc"=>20.0,"z_pk"=>"0.0,10","h"=>0.6736,"omega_b"=>0.02237,"omega_cdm"=>0.120,"ln10^{10}A_s"=>3.0363942552728806,"n_s"=>0.9649,"tau_reio"=>0.0568,"N_ur"=>2.033,"N_ncdm"=>1,"m_ncdm"=>0.06)
end
function compute_observables(p,b::Backend)
    z=0.5; h=p["H0"]/100; c=b.classy.Class(); cf=b.classy.Class()
    try
        c.set(class_params(p)); c.compute(); cf.set(fid_params()); cf.compute()
        mnu=0.06; omega_nu=0.0106*mnu; OmegaM=(p["omch2"]+0.02237+omega_nu)/h^2; fnu=c.Omega_nu/c.Omega_m
        # same approximation as legacy utils_velocileptors.py, implemented locally to avoid Python path dependence
        f=growth_f(1/(1+z),OmegaM,b.special)*(1-0.6*fnu)
        plin=[Float64(c.pk_cb(k*h,z))*h^3 for k in b.konh]
        Hfid=cf.Hubble(z)*299792.458/(cf.Hubble(0.0)/100); Hmod=c.Hubble(z)*299792.458/(c.Hubble(0.0)/100)
        DMfid=cf.angular_distance(z)*(1+z)*(cf.Hubble(0.0)/100); DMmod=c.angular_distance(z)*(1+z)*(c.Hubble(0.0)/100)
        apar=Hfid/Hmod; aperp=DMmod/DMfid
        knw,pnw=b.pnw.pnw_dst(b.konh,plin)
        m=b.rept.REPT(knw,plin;pnw=pnw,rbao=110,kmin=5e-4,kmax=0.5,nk=50,beyond_gauss=true,one_loop=true,N=2000,extrap_min=-6,extrap_max=2,cutoff=100,threads=1)
        m.compute_redshift_space_power_multipoles_tables(f;apar=apar,aperp=aperp,ngauss=4)
        r=(kv=Vector{Float64}(m.kv),pk_lin=plin,pk_0=Array(m.p0ktable),pk_2=Array(m.p2ktable),pk_4=Array(m.p4ktable),knw=knw,Pnw=pnw)
        all(x->all(isfinite,x),r)||error("REPT output contains NaN or Inf"); r
    finally; try c.struct_cleanup();c.empty();cf.struct_cleanup();cf.empty() catch end; end
end
function write_sample(root,p,r)
    text=join(("$k=$(p[k])" for k in PARAMETER_NAMES),";");dir=joinpath(root,"sample_"*bytes2hex(sha1(text))[1:16]);mkdir(dir)
    for k in (:kv,:pk_lin,:pk_0,:pk_2,:pk_4,:knw,:Pnw);npzwrite(joinpath(dir,"$k.npy"),getproperty(r,k));end
    q=Dict{String,Any}(p);q["sample_id"]=basename(dir);open(joinpath(dir,"effort_dict.json"),"w") do io;JSON3.write(io,q);end;dir
end
end
