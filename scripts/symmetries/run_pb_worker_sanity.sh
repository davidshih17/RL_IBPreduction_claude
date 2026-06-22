#!/bin/bash
# Sanity: run ONE pentagon-box worker locally exactly as the JDL will,
# with a small N_SCRAMBLES so it finishes in ~minutes. Confirms env + shell
# + python wiring before submitting the full 100-worker Condor batch.
set -u
export SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
export PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=$SAILIR_DIR/scripts/symmetries/logs/pb_worker_sanity.log

mkdir -p $SAILIR_DIR/scripts/symmetries/logs $SAILIR_DIR/scripts/symmetries/tmp_pb_sanity

WORKER_ID=0
N_SCRAMBLES=20      # small so it finishes quickly
OUTDIR=$SAILIR_DIR/scripts/symmetries/tmp_pb_sanity

PYTHONUNBUFFERED=1 bash $SAILIR_DIR/data-gen/datagen_worker_pentagonbox.sh \
    $WORKER_ID $N_SCRAMBLES $OUTDIR > $LOG 2>&1
echo "exit=$?"
echo
echo "=== log tail ==="
tail -25 $LOG
echo
echo "=== output ==="
ls -la $OUTDIR/multisector_data_worker${WORKER_ID}.jsonl 2>/dev/null
wc -l $OUTDIR/multisector_data_worker${WORKER_ID}.jsonl 2>/dev/null
