#!/bin/bash
# Condor worker for the 10× pentagon-box dataset.
# Same settings as datagen_worker_pentagonbox.sh (all sectors, no bias, same
# step/prime ranges) but seeds are OFFSET by +100M so the resulting scrambles
# are disjoint from the v2 dataset (which used seeds in [0, 100M)).
#
# Usage:
#   ./datagen_worker_pentagonbox_10x.sh <worker_id> [n_scrambles] [output_dir]
set -e

WORKER_ID=$1
N_SCRAMBLES=${2:-1000}
OUTPUT_DIR=${3:-${SAILIR_DIR}/data/pentagonbox_10x_raw_jsonl}
PYTHON=${PYTHON:-python3}

if [[ -z "${SAILIR_DIR:-}" ]]; then
    echo "ERROR: set SAILIR_DIR to the repo root before running." >&2
    exit 1
fi

cd "${SAILIR_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Disjoint seed range per worker, offset by 100M to avoid v2 overlap.
START_SEED=$((WORKER_ID * 1000000 + 100000000))
OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_worker${WORKER_ID}.jsonl"

echo "========================================"
echo "SAILIR pentagon-box 10× data-gen worker"
echo "========================================"
echo "Worker ID:    ${WORKER_ID}"
echo "Start seed:   ${START_SEED}"
echo "N scrambles:  ${N_SCRAMBLES}"
echo "Output file:  ${OUTPUT_FILE}"
echo "========================================"

PYTHONUNBUFFERED=1 "${PYTHON}" -u scripts/data_gen/generate_multisector_data.py \
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
