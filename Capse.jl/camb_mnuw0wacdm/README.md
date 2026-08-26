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
w0       [-3, 0.5]
wa       [-3, 2]
```

Design points are drawn from an oversized Latin hypercube over the rectangular
bounds and points violating `w0 + wa < -0.5` are rejected. This fills the complete
allowed two-dimensional `w0`-`wa` region without imposing an artificial
correlation between the two parameters.

Spectra use the official ACT DR6 CAMB accuracy configuration: `kmax=10`,
`k_per_logint=130`, nonlinear lensing and matter power, `lens_potential_accuracy=8`,
`lens_margin=2050`, `lAccuracyBoost=1.2`, `min_l_logl_sampling=6000`,
`DoLateRadTruncation=false`, CosmoRec, and Mead 2020 Halofit. Helium is set by
CAMB's BBN consistency relation rather than fixed by hand.

```bash
julia --project=. setup_local.jl
julia --project=. test_design.jl
python -m unittest test_camb_worker.py
julia --project=. data_generation.jl 1000
julia --project=. data_generation.jl 1000 data/pilot --seed 20260826

julia -t 8 --project=. train.jl TT
julia -t 8 --project=. train.jl TE
julia -t 8 --project=. train.jl EE
julia -t 8 --project=. train.jl PP

# Optional positive-spectrum log-target ablation
julia -t 8 --project=. train.jl TT_LOG
julia -t 8 --project=. train.jl EE_LOG

julia --project=. validate.jl TT data/camb_mnuw0wacdm_1000 artifacts/camb_mnuw0wacdm_1000/TT
```

The generated HDF5 dataset stores only the dense spectra on `ell_dense` from
`ell=2` through `ell=9000`. It does not store pre-interpolated training
targets. During training, `train.jl` reads the requested dense observable in
chunks and applies `scipy.interpolate.CubicSpline` to the stored Lobatto grid:
TT/TE/EE use 256 nodes and PP uses 192 nodes. A seeded 80/20 split is shared
by all spectra, and the interpolation configuration is recorded in the
training metadata.

## Narval generation

The Slurm launcher uses 128 one-core CAMB workers. Threading is explicitly
disabled for Julia, OpenMP, OpenBLAS, MKL, NumExpr, and Accelerate so that each
CAMB instance stays within its one-CPU allocation:

The Python interpreter embedded by PyCall must import a CAMB build linked to
CosmoRec. `setup_local.jl` verifies this and fails before generation if the
ordinary Recfast-only CAMB package is found. On Narval the tested CAMB source is
registered in PyCall's Conda environment through
`00-camb-cosmorec-local.pth`; the source and CosmoRec data directories must
remain on shared project storage.

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
