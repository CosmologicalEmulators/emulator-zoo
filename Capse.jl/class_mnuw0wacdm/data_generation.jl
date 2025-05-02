using Distributed
using NPZ
using ClusterManagers
using EmulatorsTrainer
using JSON3
using Random
using PyCall

addprocs_lsf(40; bsub_flags=`-q long -n 1 -M 14094 -e /home/mbonici/emulator-zoo/Effort.jl/mnuw0wacdm/job.err`)#this because I am using a lsf cluster. Use the appropriate one!
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, PyCall
    pars = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mν", "w0", "wa"]
    lb = [2.5, 0.8, 50.0, 0.02, 0.09, 0.02, 0.0, -3.0, -3.0,]
    ub = [3.5, 1.10, 90.0, 0.025, 0.18, 0.12, 0.5, +0.5, +2.0]

    PyCall.py"""
    import numpy as np
    from classy import Class

    # Set parameters
    def classy_function(CosmoDict):
        cosmo_params = {
            "output": "tCl pCl lCl",
            # Increase l_max for scalar modes up to 10000:
            "l_max_scalars": 10000,
            # Enable lensing (if desired):
            "lensing": "yes",                # Redshift at which to evaluate the power spectrum
            "h": CosmoDict["H0"] / 100,        # Hubble parameter
            "omega_b": CosmoDict["ombh2"],                # Baryon density parameter
            "omega_cdm": CosmoDict["omch2"],   # Cold dark matter density parameter
            "ln10^{10}A_s": CosmoDict["ln10As"], # Amplitude of the primordial power spectrum
            "n_s": CosmoDict["ns"],                     # Scalar spectral index
            "tau_reio": CosmoDict["τ"],                # Optical depth to reionization
            "N_ur": 2.033,
            "N_ncdm": 1,
            "m_ncdm": CosmoDict["Mν"],
            "use_ppf" : "yes",
            "w0_fld" : CosmoDict["w0"],
            "wa_fld" : CosmoDict["wa"],
            "fluid_equation_of_state" : "CLP",
            "cs2_fld" : 1.,
            "Omega_Lambda" : 0.,
            "Omega_scf" : 0.,
            "accurate_lensing" : 1,
            "non_linear" : "hmcode",
        }

        print("Created cosmo_params dictionary")

        # Initialize Class and compute linear power spectrum
        cosmo = Class()

        # Set the parameters
        cosmo.set(cosmo_params)
        print("Params set")
        # Compute the cosmological observables
        cosmo.compute()
        cl = cosmo.lensed_cl(lmax=10000)
        print("Cl computed")

        # The returned dictionary 'cl' contains keys like 'tt', 'ee', 'te', etc.
        # The multipole array (l) goes from 0 up to l_max (inclusive).
        ell = np.arange(len(cl['tt']))
        factor = ell*(ell+1.)/2./np.pi
        tt = 7.42715e12*(factor*cl['tt'])[2:10000]
        ee = 7.42715e12*(factor*cl['ee'])[2:10000]
        te = 7.42715e12*(factor*cl['te'])[2:10000]
        pp = (ell*(ell+1)*ell*(ell+1)*cl['pp']/2/np.pi)[2:10000]
        return tt, ee, te, pp
    """

    n = 10000
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    s_cond = [s[8, i] + s[9, i] for i in 1:n]
    s = s[:, s_cond.<0.0]
    @info size(s)

    root_dir = "/farmdisk1/mbonici/capse_class_mnuw0wacdm_" * string(n)#this is tuned to my dir, use the right one for you!

    function classy_script(CosmoDict, root_path)
        try
            rand_str = root_path * "/" * randstring(10)
            tt, ee, te, pp = py"classy_function"(CosmoDict)
            @info "EFT computed"
            if any(isnan, tt)
                @info CosmoDict
                @info P11l
                @error "There are nan values!"
            elseif any(isnan, ee)
                @error "There are nan values!"
            elseif any(isnan, te)
                @error "There are nan values!"
            else
                mkdir(rand_str)
                npzwrite(rand_str * "/TT.npy", tt)
                npzwrite(rand_str * "/EE.npy", ee)
                npzwrite(rand_str * "/TE.npy", te)
                npzwrite(rand_str * "/PP.npy", pp)

                open(rand_str * "/capse_dict.json", "w") do io
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

EmulatorsTrainer.compute_dataset(s, pars, root_dir, classy_script)

for i in workers()
    rmprocs(i)
end

exit()
