#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=$BASE/results/pentagonbox_8_5_v6/replay_expr.log
PYTHONUNBUFFERED=1 $PY $BASE/scripts/eval/replay_expr_from_disk.py \
    $BASE/results/pentagonbox_8_5_v6/work \
    '1,1,1,1,1,1,1,1,-5,0,0' > $LOG 2>&1
cat $LOG
