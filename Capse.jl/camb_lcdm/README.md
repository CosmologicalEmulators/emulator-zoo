# CAMB LCDM Capse pipeline

This local example uses the sibling `EmulatorsTrainer.jl` checkout and the
sibling `AbstractCosmologicalEmulators.jl`/`Capse.jl` packages.

Parameter order and bounds:

```text
ln10As   [2.5, 3.5]
ns       [0.85, 1.05]
tau      [0.02, 0.15]
H0       [50, 90]
omega_b  [0.02, 0.025]
omega_c  [0.08, 0.16]
```

Set up the local Julia environment:

```bash
julia --project=. setup_local.jl
```

Generate 500 CAMB cosmologies:

```bash
julia --project=. data_generation.jl 500
```

Train one or more spectra:

```bash
julia -t 8 --project=. train.jl TT
julia -t 8 --project=. train.jl TE
julia -t 8 --project=. train.jl EE
julia -t 8 --project=. train.jl PP
```

Validate an artifact against its held-out 20% split:

```bash
julia --project=. validate.jl TT data/camb_lcdm_500 artifacts/camb_lcdm_500/TT
```

TT/TE/EE use 256 Chebyshev-Lobatto nodes and linear targets. PP uses 192
Lobatto nodes and a log target. The split is reproducible through a local seed.
