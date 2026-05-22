#!/bin/bash
# Merge pentagon-box 10× per-worker JSONL shards.
set -e

OUTPUT_DIR=${1:-${SAILIR_DIR:-/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2}/data/pentagonbox_10x_raw_jsonl}
MERGED_FILE="${OUTPUT_DIR}/multisector_10x_training_data.jsonl"

N_WORKERS=$(ls -1 ${OUTPUT_DIR}/multisector_data_worker*.jsonl 2>/dev/null | wc -l)
echo "Found ${N_WORKERS} worker output files in ${OUTPUT_DIR}"
if [ ${N_WORKERS} -eq 0 ]; then
    echo "ERROR: no worker outputs in ${OUTPUT_DIR}" >&2
    exit 1
fi

cat ${OUTPUT_DIR}/multisector_data_worker*.jsonl > "${MERGED_FILE}"
TOTAL=$(wc -l < "${MERGED_FILE}")
echo "Total samples: ${TOTAL}"
echo "Merged: ${MERGED_FILE}"
