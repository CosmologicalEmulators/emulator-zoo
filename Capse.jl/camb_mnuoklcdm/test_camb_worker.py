import unittest

import numpy as np

import camb_worker


FIDUCIAL_POINT = {
    "ln10As": 3.044,
    "ns": 0.9649,
    "tau": 0.0568,
    "H0": 67.36,
    "omega_b": 0.02237,
    "omega_c": 0.12,
    "Mnu": 0.06,
    "OmegaK": 0.0,
}


class CambWorkerTests(unittest.TestCase):
    def test_act_configuration(self):
        self.assertEqual(camb_worker.OUTPUT_LMAX, 9500)
        pars = camb_worker._build_params(FIDUCIAL_POINT, 500)

        self.assertEqual(type(pars.Recomb).__name__, "CosmoRec")
        self.assertEqual(pars.lens_output_margin, 2050)
        self.assertEqual(pars.max_l, 2550)
        self.assertEqual(pars.Accuracy.AccuracyBoost, 1.0)
        self.assertEqual(pars.Accuracy.lSampleBoost, 1.0)
        self.assertEqual(pars.Accuracy.lAccuracyBoost, 1.2)
        self.assertEqual(pars.min_l_logl_sampling, 6000)
        self.assertFalse(pars.DoLateRadTruncation)
        self.assertIsNotNone(pars.bbn_predictor)
        self.assertEqual(pars.DarkEnergy.w, -1.0)
        self.assertEqual(pars.DarkEnergy.wa, 0.0)

    def test_curvature_is_propagated(self):
        for omega_k in (-0.2, 0.0, 0.2):
            with self.subTest(OmegaK=omega_k):
                point = dict(FIDUCIAL_POINT, OmegaK=omega_k)
                pars = camb_worker._build_params(point, 500)
                self.assertAlmostEqual(pars.omk, omega_k)
                self.assertEqual(pars.DarkEnergy.w, -1.0)
                self.assertEqual(pars.DarkEnergy.wa, 0.0)

    def test_boundary_spectra(self):
        for omega_k in (-0.2, 0.2):
            with self.subTest(OmegaK=omega_k):
                point = dict(FIDUCIAL_POINT, OmegaK=omega_k)
                spectra = camb_worker.compute_spectra(point, lmax=500)

                self.assertEqual(
                    set(spectra),
                    {"TT_dense", "TE_dense", "EE_dense", "BB_dense", "PP_dense"},
                )
                self.assertTrue(all(values.shape == (499,) for values in spectra.values()))
                self.assertTrue(all(np.all(np.isfinite(values)) for values in spectra.values()))
                self.assertTrue(np.all(spectra["TT_dense"] > 0.0))
                self.assertTrue(np.all(spectra["EE_dense"] > 0.0))
                self.assertTrue(np.all(spectra["BB_dense"] > 0.0))
                self.assertTrue(np.all(spectra["PP_dense"] > 0.0))


if __name__ == "__main__":
    unittest.main()
