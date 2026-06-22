#!/bin/bash
# Quick local smoke test: 30 steps on (8,4) starting integral. If this runs
# clean we'll launch a full reduction via Condor.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/v5_smoke_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO \
    --model $MODEL \
    --integral='-1,2,1,0,1,2,1,1,-3,0,0' \
    --beam-width 40 \
    --max-steps 30 \
    --max-actions 900 \
    --beam-sort mixed \
    --no-paper-masters-only \
    --prime 1009 \
    --n-threads 1 \
    --device cpu > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG (last 80) ==="
tail -80 $LOG
