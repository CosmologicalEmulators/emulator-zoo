#!/bin/bash -l
#SBATCH --job-name=pybird-train
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=pybird_train_%A_%a.out
#SBATCH --error=pybird_train_%A_%a.err
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
DATASET=${DATASET:?DATASET must point to dataset.h5}
OUTPUT=${OUTPUT:?OUTPUT must point to artifact root}
STEPS_PER_SESSION=${STEPS_PER_SESSION:-1000}
SESSIONS_PER_RATE=${SESSIONS_PER_RATE:-10}
BATCH_SIZE=${BATCH_SIZE:-512}
PREPROCESSING=${PREPROCESSING:-AsDzprec}
components=(11 loop ct 11 loop ct 11 loop ct)
multipoles=(0 0 0 2 2 2 4 4 4)
component=${components[$SLURM_ARRAY_TASK_ID]}
multipole=${multipoles[$SLURM_ARRAY_TASK_ID]}
[[ -f "$DATASET" ]] || { echo "Dataset does not exist: $DATASET" >&2; exit 1; }
unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project="$PROJECT_DIR" --startup-file=no -t "$SLURM_CPUS_PER_TASK" \
    "$PROJECT_DIR/trainer.jl" --component "$component" --multipole "$multipole" \
    --preprocessing "$PREPROCESSING" --path-input "$DATASET" --path-output "$OUTPUT" \
    --steps-per-session "$STEPS_PER_SESSION" --sessions-per-rate "$SESSIONS_PER_RATE" \
    --batch-size "$BATCH_SIZE"
