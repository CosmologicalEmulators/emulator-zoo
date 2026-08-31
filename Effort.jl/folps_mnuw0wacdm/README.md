# Folps EFT Mnu-w0-waCDM emulator

Velocileptors-style Folps EFT basis emulators. Generation uses identity AP,
the 59-point velocileptors grid, native Folps IR resummation, and no
phenomenological damping. See `DESIGN.md` for the exact 21-column basis.

```bash
julia --project=. setup_local.jl
python generate_reference.py
python test_basis.py -v

julia --project=. data_generation_local.jl 50 data/smoke_50 --processes 4 --force

for ell in 0 2 4; do
    for component in 11 loop ct; do
        julia -t 8 --project=. trainer.jl \
            --component "$component" --multipole "$ell" \
            --path-input data/smoke_50/dataset.h5 \
            --path-output artifacts/smoke_50 \
            --steps-per-session 100 --sessions-per-rate 1 --batch-size 32
    done
done
```

Runtime nuisance order:

```text
b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde,
alphashot0, alphashot2, Pshot, f0
```

Effort computes `D` and `f0`, contracts the emulated basis with
`biascombination.jl`, and applies AP after intrinsic multipole construction.
