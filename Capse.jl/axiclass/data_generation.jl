using Distributed
using NPZ
using LSFClusterManager
using EmulatorsTrainer
using JSON3
using Random
using PyCall

addprocs_lsf(50; bsub_flags=`-q long -n 1 -M 14094 -e /home/mbonici/emulator-zoo/Effort.jl/mnuw0wacdm/job.err`, exeflags="--project=/home/mbonici/emulator-zoo/Capse.jl/class_mnuw0wacdm")#this because I am using a lsf cluster. Use the appropriate one!
@info "Added processes!"
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, PyCall
    pars = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "fede", "scf", "log10axion"]
    lb = [2.5, 0.8, 50.0, 0.02, 0.09, 0.01, 1e-4, 0.0, -4.5]
    ub = [3.5, 1.10, 90.0, 0.025, 0.18, 0.15, 0.5, π, -3.0]
end

@everywhere begin
    PyCall.py"""
    import numpy as np
    import pyclass.axiclass as pya

    # Set parameters
    def axiclassy_function(CosmoDict):
        cosmo_params = {
            "output": "tCl pCl lCl",
            # Increase l_max for scalar modes up to 10000:
            "l_max_scalars": 3000,
            # Enable lensing (if desired):
            "lensing": "yes",                # Redshift at which to evaluate the power spectrum
            "h": CosmoDict["H0"] / 100,        # Hubble parameter
            "omega_b": CosmoDict["ombh2"],                # Baryon density parameter
            "omega_cdm": CosmoDict["omch2"],   # Cold dark matter density parameter
            "ln10^{10}A_s": CosmoDict["ln10As"], # Amplitude of the primordial power spectrum
            "n_s": CosmoDict["ns"],                     # Scalar spectral index
            "tau_reio": CosmoDict["τ"],                # Optical depth to reionization
            "N_ur": 2.0308,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            'fraction_axion_ac':CosmoDict["fede"],
            'scf_parameters': [CosmoDict["scf"], 0.0],
            'log10_axion_ac':CosmoDict["log10axion"],
            'do_shooting':True,
            'do_shooting_scf': True,
            'scf_potential':'axion',
            'n_axion':3,
            'security_small_Omega_scf':0.001,
            'n_axion_security': 2.09,
            'use_big_theta_scf':True,
            'scf_has_perturbations':True,
            'attractor_ic_scf':False,
            'scf_tuning_index':0
        }

        print("Created cosmo_params dictionary")

        # Initialize Class and compute linear power spectrum
        cosmo = pya.ClassEngine(EDEdict)
        hr = pya.Harmonic(cosmo)
        cl = hr.lensed_cl(lmax=3000)

        print("Cl computed")

        # The returned dictionary 'cl' contains keys like 'tt', 'ee', 'te', etc.
        # The multipole array (l) goes from 0 up to l_max (inclusive).
        ell = np.arange(len(cl['tt']))
        factor = ell*(ell+1.)/2./np.pi
        tt = 7.42715e12*(factor*cl['tt'])[2:3000]
        ee = 7.42715e12*(factor*cl['ee'])[2:3000]
        te = 7.42715e12*(factor*cl['te'])[2:3000]
        pp = (ell*(ell+1)*ell*(ell+1)*cl['pp']/2/np.pi)[2:3000]
        return tt, ee, te, pp
    """

    n = 10001
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    @info size(s)

    root_dir = "/farmdisk1/mbonici/capse_axiclass_" * string(n)#this is tuned to my dir, use the right one for you!

    function axiclassy_script(CosmoDict, root_path)
        try
            rand_str = root_path * "/" * randstring(10)
            tt, ee, te, pp = py"axiclassy_function"(CosmoDict)
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
