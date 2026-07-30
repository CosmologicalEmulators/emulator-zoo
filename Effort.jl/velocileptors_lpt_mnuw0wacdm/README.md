# CLASS + Velocileptors LPT Mnu-w0-waCDM

`generation.jl` contains the scientific model, parameter space, validation, and
sample serialization. It contains no scheduler, worker-count, queue, or machine
path configuration.

Local generation:

```bash
julia --project=. setup_local.jl
julia --project=. data_generation_local.jl \
    --samples 1000 \
    --output data/velocileptors_lpt_mnuw0wacdm_1000
```

LSF generation:

```bash
julia --project=. data_generation_lsf.jl \
    --samples 200000 \
    --output /farmdisk1/$USER/effort_velocileptors_lpt_mnuw0wacdm_200000 \
    --workers 80 \
    --queue long
```

Representative training command:

```bash
julia -t 8 --project=. trainer.jl \
    --component loop \
    --multipole 0 \
    --path-input data/velocileptors_lpt_mnuw0wacdm_1000 \
    --path-output artifacts/velocileptors_lpt_mnuw0wacdm_1000
```

Standard 50-sample smoke test:

```bash
julia --project=. smoke_test.jl
```

This generates 50 cosmologies, trains the loop monopole for 1000 optimizer
steps using a 40/10 split, loads the resulting Effort component artifact, and
requires both a finite validation loss and a finite `(59, 9)` prediction.
