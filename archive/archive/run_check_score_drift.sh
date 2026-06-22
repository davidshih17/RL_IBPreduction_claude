#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/check_score_drift_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/check_score_drift_step166.py \
    --baseline-ckpt results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0166 \
    --v4-ckpt results/probe_84_delta_v4_rescue_thick/result.pkl.ckpt.r1.step0166 \
    --topology $TOPO --model $MODEL --max-actions 900 > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG ==="
cat $LOG
