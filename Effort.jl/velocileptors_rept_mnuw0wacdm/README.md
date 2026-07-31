# Velocileptors REPT Mnu-w0-waCDM

This pipeline generates and trains component emulators for the
Velocileptors REPT model in `Mnu-w0-waCDM`. It follows the restructured
EmulatorsTrainer workflow used by the neighboring LPT pipeline:

- `generation.jl` contains the scientific sample calculation only.
- `data_generation_local.jl` and `data_generation_lsf.jl` are execution
  launchers that write worker-sharded HDF5 datasets and merge them.
- `trainer.jl` consumes `dataset.h5` and trains the `11`, `loop`, or `ct`
  component for multipole `0`, `2`, or `4`.
- `smoke_test.jl` runs the complete 50-sample local path.

## Local setup and smoke test

```bash
julia --project=. setup_local.jl
julia --project=. smoke_test.jl
```

The smoke test generates 50 candidates, uses a 40/10 train/validation split,
and trains the monopole loop component for 100 steps. Its REPT grid has 80
wavenumbers and nine loop outputs, so the final prediction shape is `80 × 9`.

## Dataset generation

Local generation:

```bash
julia --project=. data_generation_local.jl \
    --samples 1000 \
    --processes 8 \
    --output data/rept_1000
```

LSF generation:

```bash
julia --project=. data_generation_lsf.jl \
    --samples 200000 \
    --workers 80 \
    --output /farmdisk1/mbonici/effort_velocileptors_rept_mnuw0wacdm_200000
```

The merged output is:

```text
<output>/dataset.h5
```

Each sample contains `kv`, `pk_lin`, `pk_0`, `pk_2`, `pk_4`, `knw`, and
`Pnw`. Failed samples are skipped and recorded by EmulatorsTrainer's HDF5
generation metadata.

Slurm generation uses 128 one-CPU tasks and does not request a fixed node
count, allowing Slurm to place the workers across available nodes:

```bash
sbatch --account=rrg-wperciva \
    --export=ALL,PROJECT_DIR="$PWD",SAMPLES=200000 \
    rept_generate.sbatch
```

Override `OUTPUT`, `SEED`, or set `FORCE=1` through `--export` when needed.

## Training

Train one component/multipole combination with:

```bash
julia -t 8 --project=. trainer.jl \
    --component loop \
    --multipole 0 \
    --path-input data/rept_1000/dataset.h5 \
    --path-output artifacts/rept_1000 \
    --steps-per-session 2000 \
    --sessions-per-rate 10 \
    --batch-size 256
```

The artifact is written to:

```text
artifacts/rept_1000/0/loop/
```

The other supported components are `11` and `ct`; the supported multipoles
are `0`, `2`, and `4`. The trainer preserves the existing preprocessing
convention: `11` and `ct` are scaled by `A_s D²`, while `loop` is scaled by
`(A_s D²)²` before normalization.
