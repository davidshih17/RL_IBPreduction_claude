#!/bin/bash
# Independently verify the FRESH (8,5) v7 1-cpu reduction using the EXISTING
# replay pipeline (NO new replay logic):
#   1. save_replay_state.py  -- scan the fresh run's work/results/*.pkl, build the
#      cache, replay the (8,5) start integral, save replay_state.pkl (active_expr).
#   2. print_replay_terms.py -- classify every surviving term PAPER/CORNER/NON.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
FRESH=$BASE/results/pentagonbox_8_5_v7_fresh
START="1,1,1,1,1,1,1,1,-5,0,0"
LOG=$FRESH/logs/verify_fresh_replay.log

echo "[1/2] save_replay_state.py on fresh sweep_root ..." | tee $LOG
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/save_replay_state.py \
    $FRESH "$START" \
    --out $FRESH/replay_state.pkl >> $LOG 2>&1

echo "[2/2] print_replay_terms.py on the resulting replay_state ..." | tee -a $LOG
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/print_replay_terms.py \
    $FRESH/replay_state.pkl >> $LOG 2>&1

echo "DONE -> $LOG" | tee -a $LOG
