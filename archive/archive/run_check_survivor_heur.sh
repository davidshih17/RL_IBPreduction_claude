#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/check_survivor_heur_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

# Run for step 166->167 (the divergence transition)
CKPT_P=results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0166
CKPT_C=results/probe_84_delta_baseline_thick/result.pkl.ckpt.r1.step0167

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/check_survivor_heuristics.py \
    $CKPT_P $CKPT_C $TOPO > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
echo "=== LOG ==="
cat $LOG
