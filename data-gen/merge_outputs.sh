#!/bin/bash
# Optional: concatenate the per-worker JSONL shards into ONE raw file. The packed
# preprocess (Stage 2) operates per-shard and does NOT need this — merge only if
# a downstream tool wants a single raw JSONL.
#   usage:  SAILIR_DIR=... DATASET=... ./merge_outputs.sh  [raw_jsonl_dir]
set -e
RAW_DIR=${1:-${SAILIR_DIR:?set SAILIR_DIR}/data/${DATASET:?set DATASET}_raw_jsonl}
MERGED="${RAW_DIR}/multisector_training_data.jsonl"

N=$(ls -1 ${RAW_DIR}/multisector_data_worker*.jsonl 2>/dev/null | wc -l)
[ "${N}" -eq 0 ] && { echo "ERROR: no worker outputs in ${RAW_DIR}" >&2; exit 1; }

cat ${RAW_DIR}/multisector_data_worker*.jsonl > "${MERGED}"
echo "merged ${N} shards -> ${MERGED}  ($(wc -l < "${MERGED}") samples)"
