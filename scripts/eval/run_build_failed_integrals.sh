#!/bin/bash
# Launch the fast failure classifier on the (8,5) sweep work dir.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOG=$BASE/results/pentagonbox_8_5_v6/build_failed_integrals.log

cd $BASE
PYTHONUNBUFFERED=1 $PY scripts/eval/build_failed_integrals.py \
    results/pentagonbox_8_5_v6 > $LOG 2>&1
