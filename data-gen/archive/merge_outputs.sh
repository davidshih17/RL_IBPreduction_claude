#!/bin/bash
# Merge per-worker JSONL shards into a single training file and print stats.
#
# Usage:  ./merge_outputs.sh [output_dir]
# If output_dir is not given, falls back to ${SAILIR_DIR}/data/raw_jsonl.

set -e

if [[ -n "${1:-}" ]]; then
    OUTPUT_DIR=$1
elif [[ -n "${SAILIR_DIR:-}" ]]; then
    OUTPUT_DIR=${SAILIR_DIR}/data/raw_jsonl
else
    echo "ERROR: pass output_dir as \$1, or set SAILIR_DIR." >&2
    exit 1
fi

MERGED_FILE="${OUTPUT_DIR}/multisector_training_data.jsonl"

echo "========================================"
echo "Merging SAILIR training data shards"
echo "========================================"
echo "Input dir:   ${OUTPUT_DIR}"
echo "Output file: ${MERGED_FILE}"
echo "========================================"

N_WORKERS=$(ls -1 ${OUTPUT_DIR}/multisector_data_worker*.jsonl 2>/dev/null | wc -l)
echo "Found ${N_WORKERS} worker output files"
if [ ${N_WORKERS} -eq 0 ]; then
    echo "ERROR: no worker outputs in ${OUTPUT_DIR}" >&2
    exit 1
fi

cat ${OUTPUT_DIR}/multisector_data_worker*.jsonl > "${MERGED_FILE}"
TOTAL=$(wc -l < "${MERGED_FILE}")
echo "Total samples: ${TOTAL}"

${PYTHON:-python3} - "${MERGED_FILE}" <<'PY'
import json, sys
from collections import Counter
by_sector = Counter()
with open(sys.argv[1]) as f:
    for line in f:
        by_sector[json.loads(line)['sector_id']] += 1
print("Samples per sector:")
for sid in sorted(by_sector):
    print(f"  Sector {sid:2d}: {by_sector[sid]:6d}")
print(f"Total across {len(by_sector)} sectors: {sum(by_sector.values())}")
PY

echo
echo "Merged: ${MERGED_FILE}"
