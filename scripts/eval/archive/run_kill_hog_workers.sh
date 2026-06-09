#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=$BASE/results/pentagonbox_8_5_v6/kill_hog_workers.log
PYTHONUNBUFFERED=1 $PY $BASE/scripts/eval/kill_hog_workers.py \
    $BASE/results/pentagonbox_8_5_v6 --threshold-mb 8000 > $LOG 2>&1
tail -80 $LOG
