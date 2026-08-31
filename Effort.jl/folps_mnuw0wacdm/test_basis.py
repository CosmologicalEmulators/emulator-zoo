import unittest
from pathlib import Path

import numpy as np

from folps_basis import ELLS, bias_coefficients, contract_basis
from folps_worker import Backend
from generate_reference import PARAMETERS

class BiasBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = Path(__file__).resolve().parent / "reference_basis"
        cls.result, cls.table, cls.table_now = Backend().compute(PARAMETERS, return_native=True)
        cls.basis = {ell: cls.result[f"pk_{ell}"] for ell in ELLS}

    def test_shapes_and_fixtures(self):
        np.testing.assert_array_equal(self.result["k"], np.loadtxt(self.reference / "k.txt"))
        for ell in ELLS:
            self.assertEqual(self.basis[ell].shape, (59, 21))
            np.testing.assert_allclose(
                self.basis[ell], np.loadtxt(self.reference / f"pk_{ell}_basis.txt"),
                rtol=1e-11, atol=1e-10,
            )

    def test_multiple_bias_contractions_match_native_folps(self):
        cases = np.atleast_2d(np.loadtxt(self.reference / "nuisances.txt"))
        for index, nuisance in enumerate(cases):
            prediction = contract_basis(self.basis, self.result["k"], nuisance)
            reference = np.loadtxt(self.reference / f"pkell_case_{index}.txt")
            np.testing.assert_allclose(prediction, reference, rtol=2e-13, atol=3e-10)

    def test_bias_vector_has_expected_size(self):
        nuisance = np.loadtxt(self.reference / "nuisances.txt")[0]
        self.assertEqual(bias_coefficients(nuisance).shape, (21,))

    def test_decomposition_at_varied_cosmologies(self):
        from folps import RSDMultipolesPowerSpectrumCalculator
        calculator = RSDMultipolesPowerSpectrumCalculator(model="EFT")
        cases = [
            {"z": 0.3, "ln10As": 2.2, "ns": 0.85, "H0": 55.0,
             "ombh2": 0.0205, "omch2": 0.09, "Mnu": 0.15, "w0": -2.5, "wa": 1.0},
            {"z": 1.7, "ln10As": 3.3, "ns": 1.05, "H0": 85.0,
             "ombh2": 0.0245, "omch2": 0.17, "Mnu": 0.45, "w0": -0.7, "wa": -0.5},
        ]
        template = np.loadtxt(self.reference / "nuisances.txt")[0]
        for parameters in cases:
            result, table, table_now = Backend().compute(parameters, return_native=True)
            nuisance = template.copy()
            nuisance[-1] = result["f0"][0]
            basis = {ell: result[f"pk_{ell}"] for ell in ELLS}
            prediction = contract_basis(basis, result["k"], nuisance)
            native = calculator.get_rsd_pkell(
                result["k"], 1.0, 1.0, np.concatenate((nuisance[:11], [0.0])),
                table, table_now, damping=None,
            )
            np.testing.assert_allclose(prediction, native, rtol=3e-13, atol=5e-9)

if __name__ == "__main__":
    unittest.main()
