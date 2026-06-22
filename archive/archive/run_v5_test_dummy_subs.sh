#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/v5_test_dummy_subs_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/v5_test_dummy_subs.py \
    results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0166 \
    $TOPO --model $MODEL > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG ==="
cat $LOG
