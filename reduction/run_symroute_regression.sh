#!/bin/bash
# Regression: old (git HEAD) vs refactored symmetry_route demo — outputs must match.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE/reduction
cp /tmp/old_symroute.py ./_old_symroute_regression.py
{
  echo "=== OLD (git HEAD) ==="
  PYTHONUNBUFFERED=1 $PY _old_symroute_regression.py
  echo "=== NEW (refactored) ==="
  PYTHONUNBUFFERED=1 $PY symmetry_route.py
} > logs/symroute_regression_v1.log 2>&1
rm -f _old_symroute_regression.py
