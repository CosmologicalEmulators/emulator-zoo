#!/bin/bash -l
#SBATCH --job-name=rept-train
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=rept_train_%A_%a.out
#SBATCH --error=rept_train_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
DATASET=${DATASET:?DATASET must point to dataset.h5}
OUTPUT=${OUTPUT:?OUTPUT must point to the artifact root}
STEPS_PER_SESSION=${STEPS_PER_SESSION:-2000}
SESSIONS_PER_RATE=${SESSIONS_PER_RATE:-10}
BATCH_SIZE=${BATCH_SIZE:-256}

components=(11 loop ct 11 loop ct 11 loop ct)
multipoles=(0 0 0 2 2 2 4 4 4)
task_id=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}
component=${components[$task_id]}
multipole=${multipoles[$task_id]}

[[ -f "$DATASET" ]] || {
    echo "Dataset does not exist: $DATASET" >&2
    exit 1
}

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "node_list=$SLURM_JOB_NODELIST"
echo "component=$component"
echo "multipole=$multipole"
echo "dataset=$DATASET"
echo "output=$OUTPUT"
echo "steps_per_session=$STEPS_PER_SESSION"
echo "sessions_per_rate=$SESSIONS_PER_RATE"
echo "batch_size=$BATCH_SIZE"

julia --project="$PROJECT_DIR" --startup-file=no \
    -t "$SLURM_CPUS_PER_TASK" \
    "$PROJECT_DIR/trainer.jl" \
    --component "$component" \
    --multipole "$multipole" \
    --path-input "$DATASET" \
    --path-output "$OUTPUT" \
    --steps-per-session "$STEPS_PER_SESSION" \
    --sessions-per-rate "$SESSIONS_PER_RATE" \
    --batch-size "$BATCH_SIZE"
