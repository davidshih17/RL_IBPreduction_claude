#!/bin/bash
# Merge pentagon-box per-worker JSONL shards.
#
# Usage:  ./merge_outputs_pentagonbox.sh [output_dir]
set -e

if [[ -n "${1:-}" ]]; then
    OUTPUT_DIR=$1
elif [[ -n "${SAILIR_DIR:-}" ]]; then
    OUTPUT_DIR=${SAILIR_DIR}/data/pentagonbox_raw_jsonl
else
    echo "ERROR: pass output_dir as \$1, or set SAILIR_DIR." >&2
    exit 1
fi

MERGED_FILE="${OUTPUT_DIR}/multisector_training_data.jsonl"

echo "========================================"
echo "Merging pentagon-box training data shards"
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
by_t = Counter()
with open(sys.argv[1]) as f:
    for line in f:
        s = json.loads(line)
        by_sector[s['sector_id']] += 1
        # t = popcount of sector_id
        by_t[bin(s['sector_id']).count('1')] += 1
print(f"Total across {len(by_sector)} sectors")
print("Samples per t (propagator count):")
for t in sorted(by_t):
    print(f"  t={t}: {by_t[t]:8d}")
PY

echo
echo "Merged: ${MERGED_FILE}"
