#!/bin/bash -l
#SBATCH --job-name=folps-eft-10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

module load julia/1.12.5
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
FOLPS_JULIA_DEPOT=${FOLPS_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/folps_julia_depot}
SHARED_JULIA_DEPOT=${SHARED_JULIA_DEPOT:-/home/mbonici/projects/rrg-wperciva/julia_depot}
OUTPUT=${OUTPUT:?OUTPUT must be set}
SAMPLES=${SAMPLES:-10000}
PROCESSES=${PROCESSES:-31}
SEED=${SEED:-20260831}

mkdir -p "$PROJECT_DIR/logs"
unset PYTHON PYTHONHOME PYTHONPATH
export JULIA_DEPOT_PATH="$FOLPS_JULIA_DEPOT:$SHARED_JULIA_DEPOT"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_local.jl" "$SAMPLES" "$OUTPUT" \
    --processes "$PROCESSES" --seed "$SEED" --force
