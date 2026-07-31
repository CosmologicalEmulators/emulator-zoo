#!/bin/bash -l
#SBATCH --job-name=ace-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=ace_train_%j.out
#SBATCH --error=ace_train_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
DATASET=${DATASET:-$PROJECT_DIR/data/narval_candidates_500000/dataset.h5}
OUTPUT=${OUTPUT:-$PROJECT_DIR/artifacts/narval_candidates_500000}
BASIS=${BASIS:-ln10As}
STEPS_PER_SESSION=${STEPS_PER_SESSION:-4000}
SESSIONS_PER_RATE=${SESSIONS_PER_RATE:-40}
BATCH_SIZE=${BATCH_SIZE:-256}
SPLIT_SEED=${SPLIT_SEED:-20260761}
INITIALIZATION_SEED=${INITIALIZATION_SEED:-20260762}

[[ "$BASIS" == "ln10As" || "$BASIS" == "sigma8" ]] || {
    echo "BASIS must be ln10As or sigma8" >&2
    exit 1
}
[[ -f "$DATASET" ]] || {
    echo "Dataset does not exist: $DATASET" >&2
    exit 1
}

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "job_id=$SLURM_JOB_ID"
echo "node_list=$SLURM_JOB_NODELIST"
echo "basis=$BASIS"
echo "dataset=$DATASET"
echo "output=$OUTPUT"
echo "threads=$JULIA_NUM_THREADS"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/trainer.jl" \
    --basis "$BASIS" \
    --path-input "$DATASET" \
    --path-output "$OUTPUT" \
    --steps-per-session "$STEPS_PER_SESSION" \
    --sessions-per-rate "$SESSIONS_PER_RATE" \
    --batch-size "$BATCH_SIZE" \
    --split-seed "$SPLIT_SEED" \
    --initialization-seed "$INITIALIZATION_SEED"
