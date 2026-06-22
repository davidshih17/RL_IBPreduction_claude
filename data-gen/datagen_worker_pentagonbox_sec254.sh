#!/bin/bash
# Condor worker: generate one shard of pentagon-box training data
# RESTRICTED TO SECTOR 254 (L7 with D1 off) — dedicated dataset for the
# OOD regime that the v2 model can't handle.
#
# Usage:
#   ./datagen_worker_pentagonbox_sec254.sh <worker_id> [n_scrambles] [output_dir]

set -e

WORKER_ID=$1
N_SCRAMBLES=${2:-400}
OUTPUT_DIR=${3:-${SAILIR_DIR}/data/pentagonbox_sec254_raw_jsonl}
PYTHON=${PYTHON:-python3}

if [[ -z "${SAILIR_DIR:-}" ]]; then
    echo "ERROR: set SAILIR_DIR to the repo root before running." >&2
    exit 1
fi

cd "${SAILIR_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Disjoint seed range per worker.
START_SEED=$((WORKER_ID * 1000000))
OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_sec254_worker${WORKER_ID}.jsonl"

echo "========================================"
echo "SAILIR pentagon-box sec254 data-gen worker"
echo "========================================"
echo "Worker ID:    ${WORKER_ID}"
echo "Start seed:   ${START_SEED}"
echo "N scrambles:  ${N_SCRAMBLES}"
echo "Output file:  ${OUTPUT_FILE}"
echo "Restrict to:  sector_id 254 (L7 D1-off)"
echo "========================================"

PYTHONUNBUFFERED=1 "${PYTHON}" -u data-gen/generate_multisector_data.py \
    --topology topology_input/pentagonbox \
    --n_scrambles ${N_SCRAMBLES} \
    --start_seed ${START_SEED} \
    --restrict-sectors 254 \
    --output ${OUTPUT_FILE} \
    --prime 1009 \
    --min_steps 5 \
    --max_steps 25 \
    --ibp_path topology_input/pentagonbox/IBP \
    --li_path  topology_input/pentagonbox/LI

echo "Worker ${WORKER_ID} completed."
