using Distributed
using NPZ
using LSFClusterManager
using EmulatorsTrainer
using JSON3
using Random
using LinearAlgebra
using PyCall

addprocs_lsf(20; bsub_flags=`-q long -n 1 -M 4094 -e /home/mbonici/emulator-zoo/Effort.jl/velocileptors_rept_mnuOkcdm/job.err`, exeflags="--project=/home/mbonici/emulator-zoo/Effort.jl/velocileptors_rept_mnuOkcdm")#this because I am using a lsf cluster. Use the appropriate one!
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, LinearAlgebra, PyCall
    BLAS.set_num_threads(1)
    pars = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "Omega_k"]
    lb = [0.285, 2.0, 0.8, 50.0, 0.02, 0.08, 0.0, -0.2]
    ub = [1.9, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, +0.2]
    classy = pyimport("classy")
    REPT = pyimport("velocileptors.EPT.ept_fullresum_fftw")

    PyCall.py"""
    from scipy.special import hyp2f1
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelmin, argrelmax
    from scipy.fftpack import dst, idst
    from scipy.interpolate import InterpolatedUnivariateSpline as Spline

    def pnw_dst(k,p, ii_l=None,ii_r=None,extrap_min=1e-3, extrap_max=10, N=16):

        '''
        Implement the wiggle/no-wiggle split procedure from Benjamin Wallisch's thesis (arXiv:1810.02800)

        '''

        # put onto a linear grid
        ks = np.linspace( extrap_min, extrap_max, 2**N)
        lnps = Spline(k, np.log(k*p), ext=1)(ks)


        # sine transform
        dst_ps = dst(lnps)
        dst_odd = dst_ps[1::2]
        dst_even = dst_ps[0::2]

        # find the BAO regions
        if ii_l is None or ii_r is None:
            d2_even = np.gradient( np.gradient(dst_even) )
            ii_l = argrelmin(gaussian_filter(d2_even,4))[0][0]
            ii_r = argrelmax(gaussian_filter(d2_even,4))[0][1]
            #print(ii_l,ii_r)

            iis = np.arange(len(dst_odd))
            iis_div = np.copy(iis); iis_div[0] = 1.
            #cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
            cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )

            d2_odd = np.gradient( np.gradient(dst_odd) )
            ii_l = argrelmin(gaussian_filter(d2_odd,4))[0][0]
            ii_r = argrelmax(gaussian_filter(d2_odd,4))[0][1]
            #print(ii_l,ii_r)

            iis = np.arange(len(dst_odd))
            iis_div = np.copy(iis); iis_div[0] = 1.
            cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
            #cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )

        else:
            iis = np.arange(len(dst_odd))
            iis_div = np.copy(iis); iis_div[0] = 1.
            cutiis_odd = (iis > (ii_l) ) * (iis < (ii_r) )
            cutiis_even = (iis > (ii_l) ) *  (iis < (ii_r) )

        # ... and interpolate over them
        interp_odd = interp1d(iis[~cutiis_odd],(iis**2*dst_odd)[~cutiis_odd],kind='cubic')(iis)/iis_div**2
        interp_odd[0] = dst_odd[0]

        interp_even = interp1d(iis[~cutiis_even],(iis**2*dst_even)[~cutiis_even],kind='cubic')(iis)/iis_div**2
        interp_even[0] = dst_even[0]

        # Transform back
        interp = np.zeros_like(dst_ps)
        interp[0::2] = interp_even
        interp[1::2] = interp_odd

        lnps_nw = idst(interp) / 2**17

        return k, Spline(ks, np.exp(lnps_nw)/ks,ext=1)(k)
    """

    PyCall.py"""
    import numpy as np
    konhmin=1e-3; konhmax=10; nk=20000
    konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
    model_kbin_edges = np.concatenate( ([0.0005,],\
                            np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                            np.arange(0.03,0.51,0.01)) )

    kv = np.concatenate( ([0.0005,], np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True), np.arange(0.03,0.51,0.01)) )
    """

    n = 10000
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)

    root_dir = "/farmdisk1/mbonici/effort_velocileptors_rept_mnuOkcdm_" * string(n)#this is tuned to my dir, use the right one for you!

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

            f = cosmo.scale_independent_growth_factor_f(z)
            @info "computed f"

            plin = [cosmo.pk_cb(k * CosmoDict["H0"] / 100, z) * (CosmoDict["H0"] / 100)^3 for k in py"konh"]
            @info "Plin computed"

            apar = 1.0
            aperp = 1.0

            @info "Bkg computed!"
            knw, Pnw = py"""pnw_dst"""(py"konh", plin)
            @info "No wiggle computed"
            PT = REPT.REPT(knw, plin, pnw=Pnw, rbao=110, kv=py"kv", beyond_gauss=true,
                one_loop=true, N=2000, extrap_min=-6, extrap_max=2, cutoff=100, threads=1)

            @info "REPT created"

            # Get the tables
            PT.compute_redshift_space_power_multipoles_tables(f, apar=apar, aperp=aperp, ngauss=4)
            @info "REPT computed"
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
                npzwrite(rand_str * "/knw.npy", knw)
                npzwrite(rand_str * "/Pnw.npy", Pnw)

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
