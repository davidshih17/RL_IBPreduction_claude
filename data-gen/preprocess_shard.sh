#!/bin/bash
# ============================================================================
# SAILIR data-gen — STAGE 2 worker: preprocess ONE raw JSONL shard into PACKED
# tensors (.pt), using Condor file transfer (input JSONL staged into the job's
# scratch dir; the resulting shard_<id>/ copied back to the packed dir). NOT run
# by hand — invoked by submit_preprocess_batched.sh's JDL.
#
#   args:  <worker_id>
#   env:   SAILIR_DIR, TOPOLOGY, PYTHON
#   out:   shard_<id>/{train,val,test}.pt   (remapped to the packed dir by Condor)
# ============================================================================
set -e
WORKER_ID=$1
: "${SAILIR_DIR:?set SAILIR_DIR}" ; : "${TOPOLOGY:?set TOPOLOGY}"
PYTHON=${PYTHON:-python3}

INPUT_FILE=multisector_data_worker${WORKER_ID}.jsonl    # Condor stages this into CWD (scratch)
OUT_DIR=shard_${WORKER_ID}
mkdir -p "${OUT_DIR}"

echo "=== preprocess shard ${WORKER_ID} on $(hostname)  (scratch=$(pwd)) ==="
# Topology dir (~few KB) is read once from NFS; all bulk I/O stays in scratch.
PYTHONUNBUFFERED=1 "${PYTHON}" -u "${SAILIR_DIR}/data-gen/preprocess_to_tensors.py" \
    --topology   "${SAILIR_DIR}/${TOPOLOGY}" \
    --input      "${INPUT_FILE}" \
    --output_dir "${OUT_DIR}" \
    --val_split 0.1 --test_split 0.1 --seed $((42 + WORKER_ID))

echo "=== packed shard ${WORKER_ID}: ==="
ls -lh "${OUT_DIR}"/*.pt
