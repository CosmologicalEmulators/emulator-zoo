# CAMB Mnu-w0-waCDM Capse pipeline

This example uses the sibling local checkouts of `EmulatorsTrainer.jl`,
`AbstractCosmologicalEmulators.jl`, and `Capse.jl`.

Parameter order and bounds:

```text
ln10As   [2.5, 3.5]
ns       [0.85, 1.05]
tau      [0.02, 0.15]
H0       [50, 90]
omega_b  [0.02, 0.025]
omega_c  [0.08, 0.16]
Mnu      [0, 0.5]
w0       [-3, 1]
wa       [-3, 2]
```

All design points satisfy `w0 + wa < 0`. The assignment preserves the sampled
LHS marginal values.

```bash
julia --project=. setup_local.jl
julia --project=. data_generation.jl 1000

julia -t 8 --project=. train.jl TT
julia -t 8 --project=. train.jl TE
julia -t 8 --project=. train.jl EE
julia -t 8 --project=. train.jl PP

# Optional positive-spectrum log-target ablation
julia -t 8 --project=. train.jl TT_LOG
julia -t 8 --project=. train.jl EE_LOG

julia --project=. validate.jl TT data/camb_mnuw0wacdm_1000 artifacts/camb_mnuw0wacdm_1000/TT
```

TT/TE/EE use 256 Chebyshev-Lobatto nodes with linear targets. PP uses 192
Lobatto nodes with a log target. A seeded 80/20 split is shared by all spectra.

## Narval generation

The Slurm launcher uses 128 one-core CAMB workers. Threading is explicitly
disabled for Julia, OpenMP, OpenBLAS, MKL, NumExpr, and Accelerate so that each
CAMB instance stays within its one-CPU allocation:

```bash
export PROJECT_DIR=/home/mbonici/test_emu/emulator-zoo/Capse.jl/camb_mnuw0wacdm
export OUTPUT=/project/rrg-wperciva/mbonici/emulator_training/Capse/camb_mnuw0wacdm/camb_mnuw0wacdm_20000

cd "$PROJECT_DIR"
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",SAMPLES=20000,SEED=20260735,OUTPUT="$OUTPUT" \
    capse_generate.sbatch
```

The generated dataset is merged directly under `OUTPUT` with one HDF5 shard
per distributed worker and the fixed global design seed preserved in metadata.
