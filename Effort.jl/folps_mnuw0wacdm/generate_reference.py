from pathlib import Path

import numpy as np

from folps_basis import ELLS, contract_basis
from folps_worker import Backend

PARAMETERS = {"z": 0.8, "ln10As": 3.044, "ns": 0.965, "H0": 67.36,
    "ombh2": 0.02237, "omch2": 0.12, "Mnu": 0.06, "w0": -1.0, "wa": 0.0}

def nuisance_cases(f0):
    return np.asarray([
        [1.645, -0.46, -0.3685714285714286, 0.06552380952380952, 3.0, -28.9, 2.0, 0.2, 0.08, -8.1, 4719.7358, f0],
        [1.2, 0.3, -0.2, 0.1, -2.0, 5.0, -1.0, -0.4, 0.2, 3.0, 2200.0, f0],
        [2.1, -0.8, 0.4, -0.2, 4.0, -3.0, 0.5, 0.7, -0.1, 1.5, 7000.0, f0],
    ])

def main():
    from folps import RSDMultipolesPowerSpectrumCalculator
    output = Path(__file__).resolve().parent / "reference_basis"
    output.mkdir(exist_ok=True)
    result, table, table_now = Backend().compute(PARAMETERS, return_native=True)
    basis = {ell: result[f"pk_{ell}"] for ell in ELLS}
    k = result["k"]
    cases = nuisance_cases(float(result["f0"][0]))
    calculator = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    np.savetxt(output / "k.txt", k, fmt="%.17e")
    np.savetxt(output / "nuisances.txt", cases, fmt="%.17e")
    for ell in ELLS:
        np.savetxt(output / f"pk_{ell}_basis.txt", basis[ell], fmt="%.17e")
    for index, nuisance in enumerate(cases):
        contracted = contract_basis(basis, k, nuisance)
        native = calculator.get_rsd_pkell(k, 1.0, 1.0, np.concatenate((nuisance[:11], [0.0])), table, table_now, damping=None)
        np.testing.assert_allclose(contracted, native, rtol=2e-13, atol=3e-10)
        np.savetxt(output / f"pkell_case_{index}.txt", native, fmt="%.17e")
    print(f"Wrote Folps EFT bias-basis fixtures to {output}")

if __name__ == "__main__":
    main()
