#!/bin/bash -l
#SBATCH --job-name=folps-eft-train
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

module load julia/1.12.5
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
FOLPS_JULIA_DEPOT=${FOLPS_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/folps_julia_depot}
SHARED_JULIA_DEPOT=${SHARED_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/julia_depot}
DATASET=${DATASET:?DATASET must point to dataset.h5}
OUTPUT=${OUTPUT:?OUTPUT must point to artifact root}
STEPS_PER_SESSION=${STEPS_PER_SESSION:-100}
SESSIONS_PER_RATE=${SESSIONS_PER_RATE:-1}
BATCH_SIZE=${BATCH_SIZE:-512}

components=(11 loop ct 11 loop ct 11 loop ct)
multipoles=(0 0 0 2 2 2 4 4 4)
component=${components[$SLURM_ARRAY_TASK_ID]}
multipole=${multipoles[$SLURM_ARRAY_TASK_ID]}
[[ -f "$DATASET" ]] || { echo "Dataset does not exist: $DATASET" >&2; exit 1; }
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT"
unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_DEPOT_PATH="$FOLPS_JULIA_DEPOT:$SHARED_JULIA_DEPOT"
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

julia --project="$PROJECT_DIR" --startup-file=no -t "$SLURM_CPUS_PER_TASK" \
    "$PROJECT_DIR/trainer.jl" --component "$component" --multipole "$multipole" \
    --path-input "$DATASET" --path-output "$OUTPUT" \
    --steps-per-session "$STEPS_PER_SESSION" --sessions-per-rate "$SESSIONS_PER_RATE" \
    --batch-size "$BATCH_SIZE"
