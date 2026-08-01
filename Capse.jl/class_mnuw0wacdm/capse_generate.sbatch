#!/bin/bash -l
#SBATCH --job-name=capse-mnuw0wacdm
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=capse_generation_%j.out
#SBATCH --error=capse_generation_%j.err
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
SAMPLES=${SAMPLES:-200000}
SEED=${SEED:-20260752}
OUTPUT=${OUTPUT:-$PROJECT_DIR/data/capse_candidates_${SAMPLES}}
FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_ARGS+=(--force); fi
unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_slurm.jl" --samples "$SAMPLES" \
    --seed "$SEED" --output "$OUTPUT" "${FORCE_ARGS[@]}"
