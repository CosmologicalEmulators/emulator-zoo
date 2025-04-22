using Distributed
using NPZ
using ClusterManagers
using EmulatorsTrainer
using JSON3
using Random
using PyCall

addprocs_lsf(20; bsub_flags=`-q long -n 1 -M 7094`)#this because I am using a lsf cluster. Use the appropriate one!
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, PyCall
    classy = pyimport("classy")
    REPT = pyimport("velocileptors.EPT.ept_fullresum_fftw")
    pars = ["ln10As", "H0", "omch2"]
    lb = [2.6, 60., 0.08]
    ub = [3.4, 74., 0.16]

    #import numpy as np
    #from velocileptors.EPT.ept_fullresum_fftw import REPT
    #from classy import Class

    # For the wiggle no-wiggle split of Pk
    #from scipy.special import hyp2f1
    #from scipy.interpolate import interp1d
    #from scipy.ndimage import gaussian_filter
    #from scipy.signal import argrelmin, argrelmax
    #from scipy.fftpack import dst, idst
    #from scipy.interpolate import InterpolatedUnivariateSpline as Spline
    #import scipy

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
    def D_of_a(a,OmegaM=1):
        # From Stephen Chen
        return a * hyp2f1(1./3,1,11./6,-a**3/OmegaM*(1-OmegaM)) / hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))

    def f_of_a(a, OmegaM=1):
        # From Stephen Chen
        Da = D_of_a(a,OmegaM=OmegaM)
        ret = Da/a - a*(6*a**2 * (1 - OmegaM) * hyp2f1(4./3, 2, 17./6, -a**3 *  (1 - OmegaM)/OmegaM))/(11*OmegaM)/hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))
        return a * ret / Da
    """

    PyCall.py"""
    import numpy as np
    konhmin=1e-3; konhmax=10; nk=20000
    konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
    model_kbin_edges = np.concatenate( ([0.0005,],\
                            np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                            np.arange(0.03,0.51,0.01)) )

    kv = np.array([.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085,
    0.095, 0.105, 0.115, 0.125, 0.135, 0.145, 0.155, 0.165, 0.175,
    0.185, 0.195, 0.205, 0.215, 0.225, 0.235, 0.245, 0.255, 0.265,
    0.275, 0.285, 0.295, 0.305, 0.315, 0.325, 0.335, 0.345, 0.355,
    0.365, 0.375, 0.385, 0.395, 0.405, 0.415, 0.425, 0.435, 0.445,
    0.455, 0.465, 0.475, 0.485, 0.495])
    """




    n = 1000
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    #s_cond = [s[8, i]+s[9, i] for i in 1:n]
    #s = s[:, s_cond .<-1/3]
    #@info size(s)

    root_dir = "/farmdisk1/mbonici/effort_velocileptors_desi_1000"#this is tuned to my dir, use the right one for you!

    function camb_script(CosmoDict, root_path)
        try
        rand_str = root_path*"/"*randstring(10)

        z = 0.5
        cosmo_params = Dict(
                        "output" => "mPk",                   # Request the matter power spectrum (mPk)
                        "P_k_max_h/Mpc" => 20.0,             # Maximum k value (in units of h/Mpc)
                        "z_pk" => "0.0,10",                         # Redshift at which to evaluate the power spectrum
                        "h" => CosmoDict["H0"]/100,                        # Hubble parameter
                        "omega_b" => 0.02237,                  # Baryon density parameter
                        "omega_cdm" => CosmoDict["omch2"],                 # Cold dark matter density parameter
                        "ln10^{10}A_s" => CosmoDict["ln10As"],                     # Amplitude of the primordial power spectrum
                        "n_s" => 0.9649,                      # Scalar spectral index
                        "tau_reio" => 0.0568,                   # Optical depth to reionization
                        "N_ur" => 2.033,
                        "N_ncdm" => 1,
                        "m_ncdm" => 0.06
                                                                                                                                      )
        cosmo_params_fid = Dict(
                        "output" => "mPk",                   # Request the matter power spectrum (mPk)
                        "P_k_max_h/Mpc" => 20.0,             # Maximum k value (in units of h/Mpc)
                        "z_pk" => "0.0,10",                         # Redshift at which to evaluate the power spectrum
                        "h" => 0.6736,                        # Hubble parameter
                        "omega_b" => 0.02237,                  # Baryon density parameter
                        "omega_cdm" => 0.120,                 # Cold dark matter density parameter
                        "ln10^{10}A_s" => 3.0363942552728806,                     # Amplitude of the primordial power spectrum
                        "n_s" => 0.9649,                      # Scalar spectral index
                        "tau_reio" => 0.0568,                   # Optical depth to reionization
                        "N_ur" => 2.033,
                        "N_ncdm" => 1,
                        "m_ncdm" => 0.06
                                                                                                                                      )


        @info "Created Dict"
        cosmo = classy.Class()
        cosmo.set(cosmo_params)
        cosmo.compute()

        cosmo_fid = classy.Class()
        cosmo_fid.set(cosmo_params_fid)
        cosmo_fid.compute()

        @info "Class compute"
        mnu = 0.06
        omega_nu = 0.0106 * mnu
        OmegaM = (CosmoDict["omch2"] + 0.02237 + omega_nu) / (CosmoDict["H0"]/100) ^ 2
        sig8 = cosmo.sigma8()
        @info "sig8 computed"
        fnu = cosmo.Omega_nu / cosmo.Omega_m()
        f = py"f_of_a"(1 / (1+0.5); OmegaM = OmegaM)  * (1 - 0.6 * fnu)
        @info "computed f"



        plin = [cosmo.pk_cb(k * CosmoDict["H0"]/100, z) * (CosmoDict["H0"]/100) ^ 3 for k in py"konh"]
        @info "Plin computed"
        H_fid = cosmo_fid.Hubble(z) * 299792.458 / (cosmo_fid.Hubble(0.)/100)  # H(z) in km/s/Mpc
        H_model = cosmo.Hubble(z) * 299792.458/ (cosmo.Hubble(0.)/100)

        D_M_fid = cosmo_fid.angular_distance(z) * (1 + z) * (cosmo_fid.Hubble(0.)/100)
        D_M_model = cosmo.angular_distance(z) * (1 + z) * (cosmo.Hubble(0.)/100)#(1+z) is useless but I am stylish

        apar = H_fid / H_model
        aperp = D_M_model / D_M_fid
        check_AP = Dict(
        :H_fid => H_fid,
        :H_model => H_model,
        :D_M_fid => D_M_fid,
        :D_M_model => D_M_model,
        :apar => apar,
        :aperp => aperp,
        :f=> f)
        @info "Bkg computed!"
        knw, Pnw = py"""pnw_dst"""(py"konh", plin)
        @info "No wiggle computed"
        PT = REPT.REPT(knw, plin, pnw=Pnw, rbao = 110, kv=py"kv", beyond_gauss=true,
        one_loop= true, N = 2000, extrap_min=-6, extrap_max=2, cutoff = 100, threads=1)

        @info "REPT created"

        # Get the tables
        PT.compute_redshift_space_power_multipoles_tables(f, apar=1., aperp=1., ngauss = 4)
        @info "REPT computed"
        #print(PT.kv, PT.p0ktable, PT.p2ktable, PT.p4ktable)
        # Sample array
        if any(isnan, PT.p0ktable)
            @error "There are nan values!"
        elseif any(isnan, PT.p2ktable)
            @error "There are nan values!"
        elseif any(isnan, PT.p4ktable)
            @error "There are nan values!"
        else
            mkdir(rand_str)
            npzwrite(rand_str*"/kv.npy", vec(PT.kv))
            npzwrite(rand_str*"/pk_lin.npy", plin)
            npzwrite(rand_str*"/pk_0.npy", PT.p0ktable)
            npzwrite(rand_str*"/pk_2.npy", PT.p2ktable)
            npzwrite(rand_str*"/pk_4.npy", PT.p4ktable)
            npzwrite(rand_str*"/knw.npy", knw)
            npzwrite(rand_str*"/Pnw", Pnw)

            open(rand_str*"/effort_dict.json", "w") do io
                JSON3.write(io, CosmoDict)
            end
            #open(rand_str*"/check_AP_dict.json", "w") do io
            #    JSON3.write(io, check_AP)
            #end
            @info "File saved!"
        end
        catch e
        println("Something went wrong!")
        println(CosmoDict)
        end
    end

end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, camb_script)

for i in workers()
	rmprocs(i)
end
