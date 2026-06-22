#!/bin/bash
# Condor worker: generate one shard of pentagon-box training data.
# Mirrors datagen_worker.sh, but routes through the refactored
# topology-agnostic data generator with --topology topology_input/pentagonbox.
#
# Usage (from the orchestrator):
#   ./datagen_worker_pentagonbox.sh <worker_id> [n_scrambles] [output_dir]
#
# Required env vars:
#   SAILIR_DIR  -- absolute path to the SAILIR_phase2 repo root
#   PYTHON      -- python interpreter

set -e

WORKER_ID=$1
N_SCRAMBLES=${2:-100}
OUTPUT_DIR=${3:-${SAILIR_DIR}/data/pentagonbox_raw_jsonl}
PYTHON=${PYTHON:-python3}

if [[ -z "${SAILIR_DIR:-}" ]]; then
    echo "ERROR: set SAILIR_DIR to the repo root before running." >&2
    exit 1
fi

cd "${SAILIR_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Disjoint seed range per worker (matches trianglebox convention).
START_SEED=$((WORKER_ID * 1000000))
OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_worker${WORKER_ID}.jsonl"

echo "========================================"
echo "SAILIR pentagon-box data-gen worker"
echo "========================================"
echo "Worker ID:    ${WORKER_ID}"
echo "Start seed:   ${START_SEED}"
echo "N scrambles:  ${N_SCRAMBLES}"
echo "Output file:  ${OUTPUT_FILE}"
echo "Topology:     topology_input/pentagonbox"
echo "========================================"

PYTHONUNBUFFERED=1 "${PYTHON}" -u data-gen/generate_multisector_data.py \
    --topology topology_input/pentagonbox \
    --n_scrambles ${N_SCRAMBLES} \
    --start_seed ${START_SEED} \
    --output ${OUTPUT_FILE} \
    --prime 1009 \
    --min_steps 5 \
    --max_steps 25 \
    --ibp_path topology_input/pentagonbox/IBP \
    --li_path  topology_input/pentagonbox/LI

echo "Worker ${WORKER_ID} completed."
