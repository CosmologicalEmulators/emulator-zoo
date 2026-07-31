#!/bin/bash -l
#SBATCH --job-name=rept-mnuw0wacdm
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=rept_generation_%j.out
#SBATCH --error=rept_generation_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
SAMPLES=${SAMPLES:-200000}
SEED=${SEED:-20260744}
OUTPUT=${OUTPUT:-$PROJECT_DIR/data/rept_candidates_${SAMPLES}}

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_ARGS+=(--force)
fi

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "job_id=$SLURM_JOB_ID"
echo "node_list=$SLURM_JOB_NODELIST"
echo "allocated_nodes=$SLURM_JOB_NUM_NODES"
echo "candidate_samples=$SAMPLES"
echo "workers=$SLURM_NTASKS"
echo "output=$OUTPUT"

julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_slurm.jl" \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --output "$OUTPUT" \
    "${FORCE_ARGS[@]}"
