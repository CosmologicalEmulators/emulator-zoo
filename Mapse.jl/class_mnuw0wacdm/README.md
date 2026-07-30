# Mapse CLASS Mnu-w0-waCDM

This pipeline trains the two linear CLASS emulators consumed by current Mapse.jl:

```text
Pk_lin_mm  total-matter linear Pmm
Pk_lin_cb  cold+baryon linear Pcb
```

It deliberately does not generate or train nonlinear-power or boost emulators. Mapse computes Halofit or HMCode nonlinear power from these linear spectra.

The public parameter order is

```text
ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa
```

The generated spectra use physical units: `k` in `Mpc^-1` and `P(k)` in `Mpc^3`.

## Validation

```bash
julia --project=. setup_local.jl
julia --project=. test/runtests.jl
julia --project=. benchmark.jl
julia --project=. smoke_test.jl
```

The smoke test generates 50 cosmologies using two local Julia workers, merges HDF5 shards, trains both 17-component PCA emulators, loads the artifacts through Mapse, and checks finite positive predictions.

## Generation

```bash
julia --project=. data_generation_local.jl --samples 1000 --processes 8 --output data/class_1000

julia --project=. data_generation_lsf.jl --samples 250000 --workers 90 \
    --output /farmdisk1/mbonici/mapse_class_mnuw0wacdm_250000
```

## Training

```bash
julia --project=. trainer.jl --spectrum Pmm -i data/class_1000/dataset.h5 -o artifacts/class_1000
julia --project=. trainer.jl --spectrum Pcb -i data/class_1000/dataset.h5 -o artifacts/class_1000
```

