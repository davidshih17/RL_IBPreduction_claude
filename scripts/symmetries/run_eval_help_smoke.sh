#!/bin/bash
# Smoke test: each eval script's --help shows the new --topology arg and the
# script imports correctly.
set -u
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=scripts/symmetries/logs/eval_help_smoke.log
: > $LOG

for s in scripts/eval/onestep_worker.py \
         scripts/eval/beam_search.py \
         scripts/eval/replay_reduction_path.py \
         scripts/eval/hierarchical_reduction.py
do
    echo "==== $s --help ====" >> $LOG
    PYTHONUNBUFFERED=1 "$PY" -u "$s" --help >> $LOG 2>&1
    echo "exit=$?" >> $LOG
    echo >> $LOG
done

cat $LOG | grep -E '====|--topology|exit=' | head -20
