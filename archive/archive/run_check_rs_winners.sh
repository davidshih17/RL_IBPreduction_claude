#!/bin/bash
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
LOG=$LOGDIR/check_rs_winners_$(date +%s).log
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/check_rs_winners.py \
    results/probe_84_delta_baseline_thick $TOPO \
    --start-step 166 \
    --end-steps 180,200,220,240,260 > $LOG 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
wait $PID
echo "exit=$?"
cat $LOG
