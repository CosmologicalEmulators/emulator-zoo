#!/bin/bash -l
#SBATCH --job-name=folps-eft-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err

module load julia/1.12.5
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
FOLPS_JULIA_DEPOT=${FOLPS_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/folps_julia_depot}
SHARED_JULIA_DEPOT=${SHARED_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/julia_depot}
FOLPS_DATASET=${FOLPS_DATASET:?FOLPS_DATASET must point to dataset.h5}
FOLPS_ARTIFACT_ROOT=${FOLPS_ARTIFACT_ROOT:?FOLPS_ARTIFACT_ROOT must point to the artifact root}

unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_DEPOT_PATH="$FOLPS_JULIA_DEPOT:$SHARED_JULIA_DEPOT"
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FOLPS_DATASET FOLPS_ARTIFACT_ROOT

julia --project="$PROJECT_DIR" --startup-file=no -t "$SLURM_CPUS_PER_TASK" \
    "$PROJECT_DIR/check_smoke_artifacts.jl"
