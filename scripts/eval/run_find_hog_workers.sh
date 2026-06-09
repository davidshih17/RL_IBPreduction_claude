#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=$BASE/results/pentagonbox_8_5_v6/hog_workers.log
PYTHONUNBUFFERED=1 $PY $BASE/scripts/eval/find_hog_workers.py "$@" > $LOG 2>&1
cat $LOG
