# CAMB Mnu-OmegaK-LambdaCDM pipeline design

## Cosmological model

The emulator varies eight parameters:

```text
ln10As   [2.5, 3.5]
ns       [0.85, 1.05]
tau      [0.02, 0.15]
H0       [50, 90]
omega_b  [0.02, 0.025]
omega_c  [0.08, 0.16]
Mnu      [0, 0.5] eV
OmegaK   [-0.1, 0.1]
```

Dark energy is fixed to a cosmological constant, `w0=-1` and `wa=0`. CAMB
receives `OmegaK` through `CAMBparams.set_cosmology(omk=...)`; it determines
the remaining energy budget consistently rather than imposing flatness.

## Numerical configuration

Generation retains the validated Mnu-w0-wa pipeline settings: CAMB 2.0.4,
CosmoRec, BBN-consistent helium, nonlinear lensing and matter power, Mead 2020
Halofit, ACT DR6 accuracy settings, and dense TT/TE/EE/BB/PP spectra over
`ell=2:9500`.

The design is an unconstrained eight-dimensional Latin hypercube. Unlike the
Mnu-w0-wa model, it requires no early-dark-energy rejection step.

## Training grids

TT, TE, and BB use 256 Lobatto nodes over `ell=2:9500`; PP uses 192. EE uses
the hybrid grid by default: every integer `ell=2:20` followed by 256 Lobatto
nodes over `ell=20:9500`, with the duplicate endpoint removed (274 outputs).
Setting `CAPSE_EE_DENSE_LOWELL_MAX=0` restores the standard EE grid for an
ablation.

## Validation gates

1. Deterministic design and full rectangular-domain coverage.
2. Direct CAMB tests at `OmegaK=-0.1`, `0`, and `0.1`, including verification
   that `pars.omk` is propagated and dark energy remains Lambda.
3. Dense spectra must have the expected shape and finite positive TT/EE/BB/PP.
4. A 50-sample generation/training smoke test exercises the public artifact
   path before large dataset generation.

The local backend discovers the checked-out CosmoRec CAMB build at
`cmbcheb_test/tools/CAMB-cosmorec`. On another machine it can be overridden
with `CAPSE_CAMB_SOURCE`; Narval continues to use the tested PyCall Conda
environment and its registered CAMB source.
