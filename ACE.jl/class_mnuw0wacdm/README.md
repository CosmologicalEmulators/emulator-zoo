# ACE CLASS Mnu-w0-waCDM

This pipeline emulates seven CLASS and background-growth quantities in two interchangeable parameter bases.

The LHS spans `z=[0,5]`, `ln10As=[2,4]`, `ns=[0.8,1.1]`, `H0=[50,90]`,
`ombh2=[0.02,0.025]`, `omch2=[0.08,0.18]`, `Mnu=[0,1] eV`, `w0=[-3,0.5]`,
and `wa=[-3,2]`. Candidates with `w0 + wa > 0` are rejected before CLASS is run.

The `ln10As` basis takes

```text
z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa
```

and predicts

```text
sigma8, sigma8(z), r_drag, H(z), r(z), D(z), f(z)
```

The `sigma8` basis replaces the `ln10As` input with `sigma8` and predicts

```text
ln10As, sigma8(z), r_drag, H(z), r(z), D(z), f(z)
```

## Local setup and validation

```bash
julia --project=. setup_local.jl
julia --project=. test/runtests.jl
julia --project=. benchmark.jl
julia --project=. smoke_test.jl
```

The smoke test generates 50 cosmologies with two local Julia workers, merges their HDF5 shards, and trains both bases with a 40/10 split.

## Generation

Local:

```bash
julia --project=. data_generation_local.jl \
    --samples 1000 --processes 8 --output data/ace_1000
```

LSF:

```bash
julia --project=. data_generation_lsf.jl \
    --samples 300000 --workers 120 --output /farmdisk1/mbonici/ace_class_mnuw0wacdm_300000
```

Both launchers call the same scientific implementation in `generation.jl` and produce `dataset.h5` through worker-local shards.

Narval, one complete CPU node with 50,000 LHS candidates:

```bash
sbatch --account=rrg-wperciva narval_generate.sbatch
```

## Training

```bash
julia --project=. trainer.jl --basis ln10As \
    -i data/ace_1000/dataset.h5 -o artifacts/ace_1000

julia --project=. trainer.jl --basis sigma8 \
    -i data/ace_1000/dataset.h5 -o artifacts/ace_1000
```
