using Distributed
using NPZ
using ClusterManagers
using EmulatorsTrainer
using JSON3
using Random
using PyCall

addprocs_lsf(40; bsub_flags=`-q long -n 1 -M 4094 -e /home/mbonici/emulator-zoo/Effort.jl/mnuw0wacdm/job.err`)#this because I am using a lsf cluster. Use the appropriate one!
@everywhere using PyCall
@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, PyCall
    pars = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
    lb = [0.25, 2.5, 0.8, 50.0, 0.02, 0.09, 0.0, -3.0, -3,]
    ub = [2.2, 3.5, 1.10, 90.0, 0.025, 0.18, 0.5, +0.5, 2]

    PyCall.py"""
    import numpy as np
    from classy import Class
    from pybird.correlator import Correlator

    # Set parameters
    def pybird(CosmoDict):
        z = CosmoDict["z"]
        cosmo_params = {
            "output": "mPk",                   # Request the matter power spectrum (mPk)
            "P_k_max_h/Mpc": 20.0,             # Maximum k value (in units of h/Mpc)
            "z_pk": "0.0,3.",                  # Redshift at which to evaluate the power spectrum
            "h": CosmoDict["H0"] / 100,        # Hubble parameter
            "omega_b": CosmoDict["ombh2"],                # Baryon density parameter
            "omega_cdm": CosmoDict["omch2"],   # Cold dark matter density parameter
            "ln10^{10}A_s": CosmoDict["ln10As"], # Amplitude of the primordial power spectrum
            "n_s": CosmoDict["ns"],                     # Scalar spectral index
            "tau_reio": 0.0568,                # Optical depth to reionization
            "N_ur": 2.033,
            "N_ncdm": 1,
            "m_ncdm": CosmoDict["Mν"],
            "use_ppf" : "yes",
            "w0_fld" : CosmoDict["w0"],
            "wa_fld" : CosmoDict["wa"],
            "fluid_equation_of_state" : "CLP",
            "cs2_fld" : 1.,
            "Omega_Lambda" : 0.,
            "Omega_scf" : 0.
        }

        print("Created cosmo_params dictionary")

        # Initialize Class and compute linear power spectrum
        M = Class()
        M.set(cosmo_params)
        M.compute()

        # Generate logarithmic k values and compute linear power spectrum
        kk = 10 ** np.linspace(-5, 0, 200)
        pk_lin = [M.pk_cb(k * M.h(), z) * M.h()**3 for k in kk]

        # Compute growth factors
        D1 = M.scale_independent_growth_factor(z)
        f1 = M.scale_independent_growth_factor_f(z)

        print("Linear power spectrum computed")

        # Initialize Correlator
        N = Correlator()
        dk = 0.004
        kd = np.arange(0.005, 0.3, dk)

        # Set parameters for the correlator
        N.set({
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.3,
            "xdata": kd,
            "km": 0.7,
            "kr": 0.35,
            "nd": 3e-4,  # these scales control the various EFT expansions...
            "eft_basis": "eftoflss",
            "with_stoch": True,  # there are various equivalent EFT parametrizations one can choose
            "with_bias": False,
            "with_resum": True
        })  # Explicitly written for clarity

        # Compute the correlator with input power spectrum and growth factors
        N.compute({
            "kk": kk,
            "pk_lin": pk_lin,
            "f": f1
        })
        P11l = N.bird.P11l
        Ploopl = N.bird.Ploopl
        Pctl = N.bird.Pctl
        print("Correlator computation completed")
        return kk, pk_lin, kd, P11l, Ploopl, Pctl
    """

    n = 10000
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    s_cond = [s[8, i]+s[9, i] for i in 1:n]
    s = s[:, s_cond .<0.]
    @info size(s)

    root_dir = "/farmdisk1/mbonici/effort_pybird_"*string(n)*"_mnuw0wacdm"#this is tuned to my dir, use the right one for you!

    function pybird_script(CosmoDict, root_path)
        try
            rand_str = root_path * "/" * randstring(10)
            kk, Plin, kd, P11l, Ploopl, Pctl = py"pybird"(CosmoDict)
            @info "EFT computed"
            if any(isnan, P11l)
                @info CosmoDict
                @info P11l
                @error "There are nan values!"
            elseif any(isnan, Ploopl)
                @error "There are nan values!"
            elseif any(isnan, Pctl)
                @error "There are nan values!"
            else
                mkdir(rand_str)
                npzwrite(rand_str * "/kk.npy", vec(kk))
                npzwrite(rand_str * "/pk_lin.npy", Plin)
                npzwrite(rand_str * "/kd.npy", kd)
                npzwrite(rand_str * "/P11l.npy", P11l)
                npzwrite(rand_str * "/Ploopl.npy", Ploopl)
                npzwrite(rand_str * "/Pctl.npy", Pctl)

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

EmulatorsTrainer.compute_dataset(s, pars, root_dir, pybird_script)

for i in workers()
    rmprocs(i)
end
