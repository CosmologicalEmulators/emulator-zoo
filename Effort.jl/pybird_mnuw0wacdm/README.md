# Effort PyBird Mnu-w0-waCDM

This pipeline generates CLASS + PyBird EFT tables and trains component
emulators for multipoles `0`, `2`, and `4`. The scientific calculation lives
in `generation.jl`; local, LSF, and Slurm launchers all produce the same
merged HDF5 dataset.

## Dataset generation

```bash
julia --project=. setup_local.jl
julia --project=. data_generation_local.jl --samples 50 --processes 2 \
    --output data/smoke_50
```

LSF:

```bash
julia --project=. data_generation_lsf.jl --samples 250000 --workers 90 \
    --output /farmdisk1/mbonici/pybird_mnuw0wacdm_250000
```

Slurm:

```bash
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PWD",SAMPLES=250000 \
    pybird_generate.sbatch
```

The Slurm generator requests 128 one-CPU workers, 4 GB per worker, and 12
hours. It stores the `P11l`, `Ploopl`, and `Pctl` arrays in HDF5 and stores
the `kk` and `kd` grids as static HDF5 axes.

## Train all nine emulators

```bash
export PROJECT_DIR=/home/mbonici/test_emu/emulator-zoo/Effort.jl/pybird_mnuw0wacdm
export DATASET=$PROJECT_DIR/data/pybird_candidates_250000/dataset.h5
export OUTPUT=$PROJECT_DIR/artifacts/pybird_candidates_250000
cd "$PROJECT_DIR"
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",DATASET="$DATASET",OUTPUT="$OUTPUT",PREPROCESSING=AsDzprec,STEPS_PER_SESSION=1000,SESSIONS_PER_RATE=10,BATCH_SIZE=512 \
    pybird_train.sbatch
```

The training array creates one job for each combination of
`{0, 2, 4} × {11, loop, ct}` under `artifacts/<dataset>/{0,2,4}/{11,loop,ct}`.
