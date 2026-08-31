import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("FOLPS_BACKEND", "numpy")

HERE = Path(__file__).resolve().parent
DEFAULT_FOLPSD = HERE.parents[2] / "FolpsD"
FOLPSD_ROOT = Path(os.environ.get("FOLPSD_ROOT", DEFAULT_FOLPSD)).resolve()
if str(FOLPSD_ROOT) not in sys.path:
    sys.path.insert(0, str(FOLPSD_ROOT))

from classy import Class
from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator

from folps_basis import K_GRID, project_basis


K_INPUT = np.geomspace(1.0e-4, 2.0, 1000)


class Backend:
    def __init__(self):
        matrix = MatrixCalculator(A_full=True, save_dir=str(FOLPSD_ROOT / "folps" / "output_matrices"))
        self.mmatrices = matrix.get_mmatrices()

    @staticmethod
    def class_parameters(parameters):
        mnu = parameters["Mnu"]
        return {
            "output": "mPk",
            "P_k_max_h/Mpc": 20.0,
            "z_pk": "0.0,3.0",
            "h": parameters["H0"] / 100.0,
            "omega_b": parameters["ombh2"],
            "omega_cdm": parameters["omch2"],
            "ln10^{10}A_s": parameters["ln10As"],
            "n_s": parameters["ns"],
            "tau_reio": 0.0568,
            "N_ur": 2.033 if mnu > 0.0 else 3.046,
            "N_ncdm": 1 if mnu > 0.0 else 0,
            "m_ncdm": mnu,
            "use_ppf": "yes",
            "w0_fld": parameters["w0"],
            "wa_fld": parameters["wa"],
            "fluid_equation_of_state": "CLP",
            "cs2_fld": 1.0,
            "Omega_Lambda": 0.0,
            "Omega_scf": 0.0,
        }

    def compute(self, parameters, return_native=False):
        z = parameters["z"]
        h = parameters["H0"] / 100.0
        cosmology = Class()
        try:
            cosmology.set(self.class_parameters(parameters))
            cosmology.compute()
            pklin = np.asarray([cosmology.pk_cb(k * h, z) * h**3 for k in K_INPUT])
            omega_m = float(cosmology.Omega0_m())
            omega_nu = cosmology.Omega_nu
            if callable(omega_nu):
                omega_nu = omega_nu()
            kwargs = {
                "z": z,
                "h": h,
                "Omega_m": omega_m,
                "fnu": float(omega_nu) / omega_m,
                "f0": float(cosmology.scale_independent_growth_factor_f(z)),
            }
            nonlinear = NonLinearPowerSpectrumCalculator(
                mmatrices=self.mmatrices,
                kernels="fk",
                **kwargs,
            )
            nonlinear.kTout = K_GRID
            nonlinear.nk = len(K_GRID)
            table, table_now = nonlinear.calculate_loop_table(
                k=K_INPUT,
                pklin=pklin,
                cosmo=None,
                **kwargs,
            )
            basis = project_basis(table, table_now)
            result = {
                f"pk_{ell}": np.asarray(values, dtype=np.float64)
                for ell, values in basis.items()
            }
            result["k"] = np.asarray(table[0], dtype=np.float64)
            result["f0"] = np.asarray([table[-1]], dtype=np.float64)
            if return_native:
                return result, table, table_now
            return result
        finally:
            try:
                cosmology.struct_cleanup()
                cosmology.empty()
            except Exception:
                pass
