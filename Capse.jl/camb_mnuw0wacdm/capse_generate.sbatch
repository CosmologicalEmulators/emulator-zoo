#!/bin/bash -l
#SBATCH --job-name=capse-camb
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=capse_camb_generation_%j.out
#SBATCH --error=capse_camb_generation_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
SAMPLES=${SAMPLES:-20000}
SEED=${SEED:-20260735}
OUTPUT=${OUTPUT:?OUTPUT must be set}

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_PROJECT="$PROJECT_DIR"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "job_id=$SLURM_JOB_ID"
echo "tasks=$SLURM_NTASKS"
echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "samples=$SAMPLES"
echo "seed=$SEED"
echo "output=$OUTPUT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_ARGS+=(--force); fi

julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_slurm.jl" \
    --samples "$SAMPLES" --seed "$SEED" --output "$OUTPUT" \
    "${FORCE_ARGS[@]}"
