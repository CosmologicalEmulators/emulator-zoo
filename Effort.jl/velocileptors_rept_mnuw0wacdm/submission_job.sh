#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUTPUT=${OUTPUT:-$PROJECT_DIR/data/rept_200000}

bsub -P "${PROJECT:-c7}" -q "${QUEUE:-long}" \
    -o "${PROJECT_DIR}/job.out" \
    -e "${PROJECT_DIR}/job.err" \
    -n 1 \
    -M "${MEMORY_MB:-4096}" \
    -R 'span[hosts=1]' \
    env PROJECT_DIR="$PROJECT_DIR" OUTPUT="$OUTPUT" \
        SAMPLES="${SAMPLES:-200000}" WORKERS="${WORKERS:-80}" \
        QUEUE="${QUEUE:-long}" MEMORY_MB="${MEMORY_MB:-4096}" \
        SEED="${SEED:-20260744}" FORCE="${FORCE:-0}" \
        "$PROJECT_DIR/job_data_generation.sh"
