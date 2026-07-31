#!/bin/bash -l
#SBATCH --job-name=ace-class-50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=ace_class_%j.out
#SBATCH --error=ace_class_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
SAMPLES=${SAMPLES:-50000}
SEED=${SEED:-20260760}
OUTPUT=${OUTPUT:-$PROJECT_DIR/data/narval_candidates_${SAMPLES}}
NWORKERS=$((SLURM_CPUS_PER_TASK - 1))
FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_ARGS+=(--force)
fi

(( NWORKERS > 0 )) || {
    echo "At least two CPUs are required: one coordinator and one worker" >&2
    exit 1
}

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "job_id=$SLURM_JOB_ID"
echo "node_list=$SLURM_JOB_NODELIST"
echo "candidate_samples=$SAMPLES"
echo "workers=$NWORKERS"
echo "output=$OUTPUT"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_local.jl" \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --processes "$NWORKERS" \
    --output "$OUTPUT" \
    "${FORCE_ARGS[@]}"
