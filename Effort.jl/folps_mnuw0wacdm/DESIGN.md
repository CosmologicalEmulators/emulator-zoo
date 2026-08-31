# Folps EFT velocileptors-style emulator design

## Contract

Generate native undamped Folps EFT at `qpar=qper=1`, combine wiggle and
no-wiggle terms through Folps IR resummation, assimilate all explicit
`f(k)/f0` factors into basis curves, project to `ell=0,2,4`, and emulate the
resulting multipole basis tables. Effort computes `D` and `f0`, contracts the
tables with nuisance polynomials, and applies AP at runtime.

## Grid and shapes

The output grid is exactly the 59-point velocileptors grid. For each multipole:

| component | columns | shape | amplitude preprocessing |
|---|---:|---:|---|
| `11` | 3 | `(59,3)` | divide by `As*D^2` |
| `loop` | 12 | `(59,12)` | divide by `(As*D^2)^2` |
| `ct` standard | 3 | first half of `(59,6)` | divide by `As*D^2` |
| `ct` NLO | 3 | second half of `(59,6)` | divide by `(As*D^2)^3` |

The HDF5 observables `pk_0`, `pk_2`, and `pk_4` each have shape `(59,21)`.
Nine component emulators are trained: three components for each of three
multipoles. Total learned outputs are `3*59*21 = 3717`; the largest network
has 708 outputs.

## Polynomial contraction

The 12 loop monomials are

```text
1, b1, b1^2, b2, b1*b2, b2^2,
bs, b1*bs, b2*bs, bs^2, b3, b1*b3
```

The complete learned-table coefficient vector is

```text
b1^2, 2*b1*f0, f0^2,
<12 loop monomials>,
alpha0, alpha2, alpha4,
ctilde*b1^2, 2*ctilde*b1*f0, ctilde*f0^2
```

Two stochastic basis columns are analytic and are not emulated. Their
coefficients are `Pshot*alphashot0` and `Pshot*alphashot2`.

## Exact decomposition

`folps_basis.py` reproduces the native Folps formulas term by term. It combines
wiggle/no-wiggle tables with the same IR exponential and uses the same
six-positive-node Gauss-Legendre projection as `get_rsd_pkell`. Text fixtures
freeze the 21 basis columns and native multipoles for three distinct nuisance
vectors. Contracted basis tables agree with native Folps EFT at roughly
machine precision.
