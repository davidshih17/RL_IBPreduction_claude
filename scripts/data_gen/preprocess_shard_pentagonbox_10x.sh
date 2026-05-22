#!/bin/bash
# Preprocess ONE worker file (= one shard) of the 10x pentagon-box dataset.
# Uses Condor file transfer: Condor places the input JSONL into the per-job
# scratch dir and copies the resulting shard_${WORKER_ID}/ back to the
# submit dir afterwards. We do NOT read or write the shared filesystem
# directly during the job — only at submit / completion via Condor.
set -e

WORKER_ID=$1
if [[ -z "$WORKER_ID" ]]; then
    echo "Usage: $0 <WORKER_ID>" >&2
    exit 1
fi

# We are inside the Condor per-job scratch dir at this point. Condor has
# already staged the input JSONL into the CWD.
INPUT_FILE=multisector_data_worker${WORKER_ID}.jsonl
OUT_DIR=shard_${WORKER_ID}

mkdir -p $OUT_DIR

echo "=== preprocess_shard SHARD ${WORKER_ID} ==="
echo "PWD (scratch):    $(pwd)"
echo "Scratch (Condor): ${_CONDOR_SCRATCH_DIR:-not-set}"
echo "Input file:       ${INPUT_FILE} ($(du -sh ${INPUT_FILE} | cut -f1))"
echo "Output dir:       ${OUT_DIR}"
echo "Host:             $(hostname)"
date

# Run the preprocess with relative paths so all I/O stays in scratch.
# The topology dir is small (~few KB), read once from NFS — acceptable.
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2

PYTHONUNBUFFERED=1 ${PYTHON} -u ${SAILIR_DIR}/scripts/data_gen/preprocess_to_tensors.py \
    --topology ${SAILIR_DIR}/topology_input/pentagonbox \
    --input "${INPUT_FILE}" \
    --output_dir "${OUT_DIR}" \
    --val_split 0.1 --test_split 0.1 --seed $((42 + WORKER_ID))

echo
echo "=== Done. Packed sizes (will be transferred back to submit dir): ==="
ls -lh ${OUT_DIR}/*.pt 2>&1
date
