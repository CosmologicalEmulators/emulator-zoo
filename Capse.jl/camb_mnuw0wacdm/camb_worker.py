import numpy as np

import camb


RECOMBINATION_MODEL = "CosmoRec"
LENS_MARGIN = 2050
OUTPUT_LMAX = 9500


def backend_configuration():
    probe = camb.set_params(recombination_model=RECOMBINATION_MODEL)
    model = type(probe.Recomb).__name__
    if model != RECOMBINATION_MODEL:
        raise RuntimeError(
            f"CAMB loaded recombination model {model}, expected {RECOMBINATION_MODEL}"
        )
    return {
        "camb_version": camb.__version__,
        "camb_path": camb.__file__,
        "recombination_model": model,
        "helium_fraction": "BBN consistency",
        "lens_margin": LENS_MARGIN,
        "output_lmax": OUTPUT_LMAX,
    }


def lobatto_nodes(n_nodes, lower=2.0, upper=OUTPUT_LMAX):
    theta = np.linspace(0.0, np.pi, n_nodes)
    nodes = 0.5 * (lower + upper) - 0.5 * (upper - lower) * np.cos(theta)
    nodes[0] = lower
    nodes[-1] = upper
    return nodes


def _build_params(parameters, lmax):
    mnu = parameters["Mnu"]
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=parameters["H0"],
        ombh2=parameters["omega_b"],
        omch2=parameters["omega_c"],
        omk=0.0,
        TCMB=2.7255,
        mnu=mnu,
        num_massive_neutrinos=1 if mnu > 0.0 else 0,
        nnu=3.046,
    )
    pars.InitPower.set_params(
        As=np.exp(parameters["ln10As"]) * 1.0e-10,
        ns=parameters["ns"],
        pivot_scalar=0.05,
    )
    pars.Reion.set_tau(parameters["tau"])
    pars.set_dark_energy(
        w=parameters["w0"],
        wa=parameters["wa"],
        dark_energy_model="ppf",
    )
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version="mead2020")
    pars = camb.set_params(
        cp=pars,
        recombination_model=RECOMBINATION_MODEL,
        kmax=10,
        k_per_logint=130,
        lens_potential_accuracy=8,
        lens_output_margin=LENS_MARGIN,
        AccuracyBoost=1.0,
        lSampleBoost=1.0,
        lAccuracyBoost=1.2,
        min_l_logl_sampling=6000,
        DoLateRadTruncation=False,
        halofit_version="mead2020",
        lmax=lmax,
    )
    return pars


def compute_spectra(parameters, lmax=OUTPUT_LMAX):
    pars = _build_params(parameters, lmax)
    results = camb.get_results(pars)
    cmb = results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)
    lens = results.get_lens_potential_cls(lmax=lmax, raw_cl=True)
    ell_full = np.arange(lmax + 1, dtype=np.float64)
    mask = ell_full >= 2
    ell = ell_full[mask]
    dl = ell_full * (ell_full + 1.0) / (2.0 * np.pi)
    pp_factor = (ell_full * (ell_full + 1.0)) ** 2 / (2.0 * np.pi)
    dense = {
        "TT": (cmb[: lmax + 1, 0] * dl)[mask],
        "TE": (cmb[: lmax + 1, 3] * dl)[mask],
        "EE": (cmb[: lmax + 1, 1] * dl)[mask],
        "BB": (cmb[: lmax + 1, 2] * dl)[mask],
        "PP": (lens[: lmax + 1, 0] * pp_factor)[mask],
    }
    return {f"{name}_dense": values for name, values in dense.items()}
