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

The Lux addition can be exercised locally with a 500-cosmology fixture:

```bash
julia --project=. lux_smoke_test.jl
```

This creates the ignored local files under `data/local_500/` and trains both
bases with the additional `trainer_lux.jl` path. The existing
`trainer.jl`/SimpleChains path is unchanged.

To compare Zygote with CPU Reactant after Reactant's warmup compilation, run:

```bash
julia --project=. lux_training_benchmark.jl ln10As
```

The benchmark uses 100 optimizer steps per session and 10 sessions per
learning rate, for all ten learning rates. Reactant receives one warmup step;
the reported training time excludes that warmup. Use `sigma8` as the optional
argument to benchmark the other basis.

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

All launchers call the same scientific implementation in `generation.jl` and
produce `dataset.h5` through worker-local shards.

Narval uses `SlurmClusterManager.jl` with 128 single-CPU workers, 4 GB per
worker, and 500,000 LHS candidates. The launcher deliberately omits a node
count, allowing Slurm to pack or spread the workers across available nodes:

```bash
sbatch --account=rrg-wperciva --export=ALL,FORCE=1 narval_generate.sbatch
```

CLASS failures are rejected per sample and recorded in `generation_failures.json`.
The merged dataset contains only successful samples; `generation_metadata.json`
records candidate, retained, successful, and failed counts.

## Training

The trainer uses five hidden layers of 64 neurons with `tanh` activation.
Submit one Slurm job per basis:

```bash
sbatch --account=rrg-wperciva --export=ALL,BASIS=ln10As narval_train.sbatch
sbatch --account=rrg-wperciva --export=ALL,BASIS=sigma8 narval_train.sbatch
```

The default schedule is 40 sessions at each of 10 learning rates, with 4,000
steps per session and batch size 256. Override these through `sbatch` exports,
for example `--export=ALL,BASIS=ln10As,SESSIONS_PER_RATE=1,STEPS_PER_SESSION=100`
for a short training check.

`narval_train.sbatch` requests one node, one task, 32 CPUs, 16 GB of memory,
and 12 hours of wall time. It sets `JULIA_NUM_THREADS` to the allocated CPU
count while keeping BLAS and OpenMP at one thread to avoid oversubscription.

```bash
julia --project=. trainer.jl --basis ln10As \
    -i data/ace_1000/dataset.h5 -o artifacts/ace_1000

julia --project=. trainer.jl --basis sigma8 \
    -i data/ace_1000/dataset.h5 -o artifacts/ace_1000
```

## Independent validation

`validate.jl` evaluates the entire supplied HDF5 dataset. It deliberately
does not use `validation_indices.npy`; the dataset is treated as independent
of training. It writes the 64th, 95th, and 99th percentile absolute relative
residuals for each of the seven ACE outputs, without making plots.

```bash
julia --project=. validate.jl \
    --dataset data/local_500/dataset.h5 \
    --artifact artifacts/local_500_lux/ln10As \
    --basis ln10As \
    --output validation/local_500_lux/ln10As

julia --project=. validate.jl \
    --dataset data/local_500/dataset.h5 \
    --artifact artifacts/local_500_lux/sigma8 \
    --basis sigma8 \
    --output validation/local_500_lux/sigma8
```

Each result is written to `residuals_percentiles.npy` with shape `(3, 7)`.
The output order is `sigma8`, `sigma8_z`, `r_drag`, `H_z`, `r_z`, `D_z`,
`f_z`.
