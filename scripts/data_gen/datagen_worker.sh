#!/bin/bash
# Condor worker: generate one shard of self-supervised training data.
# Reproduces the per-worker invocation used for the paper (max_steps=25,
# 100 scrambles per worker, non-overlapping seeds).
#
# Usage (from the orchestrator): ./datagen_worker.sh <worker_id> [n_scrambles] [output_dir]
#
# Required env vars:
#   SAILIR_DIR  -- absolute path to the SAILIR repo root
#   PYTHON      -- python interpreter (defaults to system python)

set -e

WORKER_ID=$1
N_SCRAMBLES=${2:-100}
OUTPUT_DIR=${3:-${SAILIR_DIR}/data/raw_jsonl}
PYTHON=${PYTHON:-python3}

if [[ -z "${SAILIR_DIR:-}" ]]; then
    echo "ERROR: set SAILIR_DIR to the repo root before running." >&2
    exit 1
fi

cd "${SAILIR_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Each worker reserves a disjoint seed range to guarantee independence across
# parallel Condor workers.
START_SEED=$((WORKER_ID * 1000000))
OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_worker${WORKER_ID}.jsonl"

echo "========================================"
echo "SAILIR data-generation worker"
echo "========================================"
echo "Worker ID:    ${WORKER_ID}"
echo "Start seed:   ${START_SEED}"
echo "N scrambles:  ${N_SCRAMBLES}"
echo "Output file:  ${OUTPUT_FILE}"
echo "========================================"

PYTHONUNBUFFERED=1 "${PYTHON}" scripts/data_gen/generate_multisector_data.py \
    --n_scrambles ${N_SCRAMBLES} \
    --start_seed ${START_SEED} \
    --output ${OUTPUT_FILE} \
    --prime 1009 \
    --min_steps 5 \
    --max_steps 25 \
    --ibp_path scripts/data_gen/IBP \
    --li_path  scripts/data_gen/LI

echo "Worker ${WORKER_ID} completed."
