using Distributed
using NPZ
using LSFClusterManager
using EmulatorsTrainer
using JSON3
using Random
using LinearAlgebra
using PyCall

addprocs_lsf(80; bsub_flags=`-q long -n 1 -M 4094 -e /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuOkcdm/job.err`, exeflags="--project=/home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuOkcdm")#this because I am using a lsf cluster. Use the appropriate one!
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, LinearAlgebra, PyCall
    BLAS.set_num_threads(1)
    pars = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "Omega_k"]
    lb = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -0.2]
    ub = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, +0.2]
    classy = pyimport("classy")
    LPT_RSD = pyimport("velocileptors.LPT.lpt_rsd_fftw")

    PyCall.py"""
    import numpy as np
    konhmin=1e-3; konhmax=10; nk=20000
    konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
    model_kbin_edges = np.concatenate( ([0.0005,],\
                            np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                            np.arange(0.03,0.51,0.01)) )

    kv = np.concatenate( ([0.0005,], np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True), np.arange(0.03,0.51,0.01)) )
    """

    n = 200000
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    @info size(s)

    root_dir = "/farmdisk1/mbonici/effort_velocileptors_lpt_mnuOkcdm_" * string(n)#this is tuned to my dir, use the right one for you!

    function velocileptors_script(CosmoDict, root_path)
        try
            rand_str = root_path * "/" * randstring(10)

            z = CosmoDict["z"]
            cosmo_params = Dict(
                "output" => "mPk",
                "P_k_max_h/Mpc" => 20.0,
                "z_pk" => "0.0,3.",
                "h" => CosmoDict["H0"] / 100,
                "omega_b" => CosmoDict["ombh2"],
                "omega_cdm" => CosmoDict["omch2"],
                "ln10^{10}A_s" => CosmoDict["ln10As"],
                "n_s" => CosmoDict["ns"],
                "tau_reio" => 0.0568,
                "N_ur" => 2.033,
                "N_ncdm" => 1,
                "m_ncdm" => CosmoDict["Mν"],
                "use_ppf" => "yes",
                "w0_fld" => -1.0,
                "wa_fld" => 0.0,
                "fluid_equation_of_state" => "CLP",
                "cs2_fld" => 1.0,
                "Omega_k" => CosmoDict["Omega_k"],
                "Omega_Lambda" => 0.0,
                #"Omega_scf" => 0.0
            )

            @info "Created Dict"
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()

            @info "Class compute"
            f = cosmo.scale_independent_growth_factor_f(z)
            @info "computed f"

            plin = [cosmo.pk_cb(k * CosmoDict["H0"] / 100, z) * (CosmoDict["H0"] / 100)^3 for k in py"konh"]
            @info "Plin computed"

            @info "Bkg computed!"
            PT = LPT_RSD.LPT_RSD(py"konh", plin, kIR=0.2, use_Pzel=false,
                cutoff=10, extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5)

            @info "LPT created"

            # Get the tables
            PT.make_pltable(f, kv=py"kv", apar=1.0, aperp=1.0, ngauss=3)
            @info "LPT computed"
            # Sample array
            if any(isnan, PT.p0ktable)
                @error "There are nan values!"
            elseif any(isnan, PT.p2ktable)
                @error "There are nan values!"
            elseif any(isnan, PT.p4ktable)
                @error "There are nan values!"
            else
                mkdir(rand_str)
                npzwrite(rand_str * "/kv.npy", vec(PT.kv))
                npzwrite(rand_str * "/pk_lin.npy", plin)
                npzwrite(rand_str * "/pk_0.npy", PT.p0ktable)
                npzwrite(rand_str * "/pk_2.npy", PT.p2ktable)
                npzwrite(rand_str * "/pk_4.npy", PT.p4ktable)

                open(rand_str * "/effort_dict.json", "w") do io
                    JSON3.write(io, CosmoDict)
                end
                @info "File saved!"
            end
        catch e
            println("Something went wrong!")
            println(CosmoDict)
        end
    end

end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, velocileptors_script)

for i in workers()
    rmprocs(i)
end
