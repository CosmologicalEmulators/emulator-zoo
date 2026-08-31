# CAMB Mnu-OmegaK-LambdaCDM Capse pipeline

This pipeline varies `ln10As`, `ns`, `tau`, `H0`, `omega_b`, `omega_c`,
`Mnu`, and `OmegaK`. It fixes `w0=-1` and `wa=0`; `OmegaK` spans `[-0.1, 0.1]`.
See `DESIGN.md` for the complete model and numerical contract.

```bash
julia --project=. setup_local.jl
julia --project=. test_design.jl
julia --project=. test_training_grid.jl
python -m unittest test_camb_worker.py

# Local generation
julia --project=. data_generation.jl 50 data/smoke_50

# Training: hybrid EE is the default in this pipeline
julia -t 8 --project=. train.jl TT_LOG data/smoke_50/dataset.h5 artifacts/smoke_50
julia -t 8 --project=. train.jl TE data/smoke_50/dataset.h5 artifacts/smoke_50
julia -t 8 --project=. train.jl EE_LOG data/smoke_50/dataset.h5 artifacts/smoke_50
julia -t 8 --project=. train.jl BB_LOG data/smoke_50/dataset.h5 artifacts/smoke_50
julia -t 8 --project=. train.jl PP data/smoke_50/dataset.h5 artifacts/smoke_50

# End-to-end 50-sample smoke test
julia --project=. smoke_test.jl
```

## Narval generation

```bash
export PROJECT_DIR=/home/mbonici/test_emu/emulator-zoo/Capse.jl/camb_mnuoklcdm
export OUTPUT=/project/rrg-wperciva/mbonici/emulator_training/Capse/camb_mnuoklcdm/camb_mnuoklcdm_omk01_20000

cd "$PROJECT_DIR"
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",SAMPLES=20000,SEED=20260735,OUTPUT="$OUTPUT" \
    capse_generate.sbatch
```
