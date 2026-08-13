# Mapse CLASS Mnu-w0-waCDM

This pipeline trains the two linear CLASS emulators consumed by current Mapse.jl:

```text
Pk_lin_mm  total-matter linear Pmm
Pk_lin_cb  cold+baryon linear Pcb
```

It deliberately does not generate or train nonlinear-power or boost emulators. Mapse computes Halofit or HMCode nonlinear power from these linear spectra.

The public prediction parameter order is

```text
ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa
```

Generation fixes `ln10As=3.044` and `ns=0.965`; the Latin hypercube spans
only `z, H0, ombh2, omch2, Mnu, w0, wa`. The trainer divides those fixed
spectra by the primordial spectrum, growth, and analytic LCDM transfer
baseline. Mapse restores arbitrary public `ln10As` and `ns` analytically at
prediction time.

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

Slurm generation:

```bash
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PWD",SAMPLES=250000 \
    mapse_generate.sbatch
```

The Slurm generator requests 128 one-CPU workers, 4 GB per worker, and 12
hours. It does not request a fixed node count.

## Training

```bash
julia --project=. trainer.jl --spectrum Pmm -i data/class_1000/dataset.h5 -o artifacts/class_1000
julia --project=. trainer.jl --spectrum Pcb -i data/class_1000/dataset.h5 -o artifacts/class_1000
```

To train both linear spectra automatically on Slurm:

```bash
export PROJECT_DIR=/home/mbonici/test_emu/emulator-zoo/Mapse.jl/class_mnuw0wacdm
export DATASET=$PROJECT_DIR/data/mapse_candidates_250000/dataset.h5
export OUTPUT=$PROJECT_DIR/artifacts/mapse_candidates_250000
cd "$PROJECT_DIR"
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",DATASET="$DATASET",OUTPUT="$OUTPUT",STEPS_PER_SESSION=1000,SESSIONS_PER_RATE=10,BATCH_SIZE=512 \
    mapse_train.sbatch
```

This creates two array jobs, one for `Pmm` and one for `Pcb`.
