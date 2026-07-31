#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SAMPLES=${SAMPLES:-200000}
WORKERS=${WORKERS:-80}
QUEUE=${QUEUE:-long}
MEMORY_MB=${MEMORY_MB:-4096}
SEED=${SEED:-20260744}
OUTPUT=${OUTPUT:-$PROJECT_DIR/data/rept_${SAMPLES}}

ARGS=(
    --samples "$SAMPLES"
    --workers "$WORKERS"
    --queue "$QUEUE"
    --memory-mb "$MEMORY_MB"
    --seed "$SEED"
    --output "$OUTPUT"
)
if [[ "${FORCE:-0}" == "1" ]]; then
    ARGS+=(--force)
fi

exec julia --project="$PROJECT_DIR" --startup-file=no \
    "$PROJECT_DIR/data_generation_lsf.jl" "${ARGS[@]}"
