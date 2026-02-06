#!/bin/bash
# Merge worker outputs into single training file and compute statistics
# Usage: ./merge_outputs.sh [output_dir]

OUTPUT_DIR=${1:-/het/p4/dshih/jet_images-deep_learning/RL_IBPreduction_claude/data/multisector}
MERGED_FILE=${OUTPUT_DIR}/multisector_training_data.jsonl

echo "========================================"
echo "Merging Multi-sector Training Data"
echo "========================================"
echo "Input directory: ${OUTPUT_DIR}"
echo "Output file: ${MERGED_FILE}"
echo "========================================"

# Count worker files
N_WORKERS=$(ls -1 ${OUTPUT_DIR}/multisector_data_worker*.jsonl 2>/dev/null | wc -l)
echo "Found ${N_WORKERS} worker output files"

if [ ${N_WORKERS} -eq 0 ]; then
    echo "ERROR: No worker output files found!"
    exit 1
fi

# Merge all worker outputs
echo "Merging files..."
cat ${OUTPUT_DIR}/multisector_data_worker*.jsonl > ${MERGED_FILE}

# Count total samples
TOTAL_SAMPLES=$(wc -l < ${MERGED_FILE})
echo "Total samples: ${TOTAL_SAMPLES}"

# Count unique scrambles
N_SCRAMBLES=$(python3 -c "
import json
scrambles = set()
with open('${MERGED_FILE}') as f:
    for line in f:
        d = json.loads(line)
        scrambles.add((d['scramble_id'], d['sector_id']))
print(len(scrambles))
")
echo "Unique scrambles: ${N_SCRAMBLES}"

# Count samples per sector
echo ""
echo "Samples per sector:"
python3 -c "
import json
from collections import Counter
sector_counts = Counter()
with open('${MERGED_FILE}') as f:
    for line in f:
        d = json.loads(line)
        sector_counts[d['sector_id']] += 1
for sid in sorted(sector_counts.keys()):
    print(f'  Sector {sid:2d}: {sector_counts[sid]:6d} samples')
print(f'  Total: {sum(sector_counts.values())} samples across {len(sector_counts)} sectors')
"

echo ""
echo "Merged file: ${MERGED_FILE}"
echo "Done!"
