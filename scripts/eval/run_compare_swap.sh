#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/compare_swap_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/compare_v4_vs_baseline_swap.py \
    --baseline-dir results/probe_84_delta_baseline_thick \
    --v4-dir results/probe_84_delta_v4_rescue_thick \
    --topology $TOPO > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG ==="
cat $LOG
