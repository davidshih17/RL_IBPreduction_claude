#!/bin/bash
# Merge per-worker JSONL shards into a single training file and print stats.
#
# Usage:  ./merge_outputs.sh [output_dir]
#
# Required env vars:
#   SAILIR_DIR  -- absolute path to the SAILIR repo root

set -e

if [[ -z "${SAILIR_DIR:-}" ]]; then
    echo "ERROR: set SAILIR_DIR to the repo root before running." >&2
    exit 1
fi

OUTPUT_DIR=${1:-${SAILIR_DIR}/data/raw_jsonl}
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
scrambles = set()
by_sector = Counter()
with open(sys.argv[1]) as f:
    for line in f:
        d = json.loads(line)
        scrambles.add((d['scramble_id'], d['sector_id']))
        by_sector[d['sector_id']] += 1
print(f"Unique scrambles: {len(scrambles)}")
print(f"Samples per sector:")
for sid in sorted(by_sector):
    print(f"  Sector {sid:2d}: {by_sector[sid]:6d}")
print(f"Total across {len(by_sector)} sectors: {sum(by_sector.values())}")
PY

echo
echo "Merged: ${MERGED_FILE}"
