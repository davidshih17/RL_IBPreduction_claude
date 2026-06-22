#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/find_picked_action_$(date +%s).log
CKPT166=results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0166
CKPT167=results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0167
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/find_picked_action_idx.py \
    $CKPT166 $CKPT167 $TOPO > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG ==="
cat $LOG
