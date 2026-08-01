# Capse CLASS Mnu-w0-waCDM

This pipeline generates and trains Capse emulators for `TT`, `TE`, `EE`, and
`PP`. `generation.jl` contains the scientific CLASS calculation; the local,
LSF, and Slurm launchers all write the same merged HDF5 dataset.

## Setup

```bash
julia --project=. setup_local.jl
julia --project=. smoke_test.jl
```

## Dataset generation

```bash
julia --project=. data_generation_local.jl --samples 1000 --processes 8 \
    --output data/capse_1000

julia --project=. data_generation_lsf.jl --samples 200000 --workers 80 \
    --output /farmdisk1/mbonici/capse_class_mnuw0wacdm_200000
```

On Slurm:

```bash
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PWD",SAMPLES=200000 \
    capse_generate.sbatch
```

The Slurm generator requests 128 one-CPU workers, 4 GB per worker, and 12
hours. It does not request a fixed node count.

## Train all four spectra

```bash
export PROJECT_DIR=/home/mbonici/test_emu/emulator-zoo/Capse.jl/class_mnuw0wacdm
export DATASET=$PROJECT_DIR/data/capse_candidates_200000/dataset.h5
export OUTPUT=$PROJECT_DIR/artifacts/capse_candidates_200000
cd "$PROJECT_DIR"
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",DATASET="$DATASET",OUTPUT="$OUTPUT",STEPS_PER_SESSION=1000,SESSIONS_PER_RATE=10,BATCH_SIZE=512 \
    capse_train.sbatch
```

The training array creates four jobs, one for each of `TT`, `TE`, `EE`, and
`PP`, under `artifacts/capse_candidates_200000/{TT,TE,EE,PP}/`.

## Local independent validation workflow

Create the training and independent validation datasets separately:

```bash
julia --project=. data_generation_local.jl \
    --samples 1000 --processes 8 --output data/local_1000

julia --project=. data_generation_local.jl \
    --samples 200 --processes 8 --output data/validation_200
```

Train all four spectra on the 1000-sample dataset:

```bash
julia --project=. train_local_all.jl \
    --dataset data/local_1000/dataset.h5 \
    --output artifacts/local_1000 \
    --steps-per-session 1000 \
    --sessions-per-rate 10 \
    --batch-size 512
```

Evaluate the full independent 200-sample dataset. This uses no training
split and writes the 64th, 95th, and 99th percentile absolute relative
residuals without making plots:

```bash
julia --project=. validate.jl \
    --dataset data/validation_200/dataset.h5 \
    --artifacts artifacts/local_1000 \
    --output validation/local_1000_on_validation_200
```

The resulting arrays are written to
`validation/local_1000_on_validation_200/{TT,TE,EE,PP}/residuals_percentiles.npy`.

For a single spectrum and its plot:

```bash
julia --project=. validate.jl \
    --dataset data/validation_200/dataset.h5 \
    --artifacts artifacts/local_1000 \
    --spectrum TT \
    --output validation/local_1000_on_validation_200

julia --project=. plot_validation.jl \
    --spectrum TT \
    --residuals validation/local_1000_on_validation_200/TT/residuals_percentiles.npy \
    --output validation/local_1000_on_validation_200/TT/residuals.png
```
