#!/bin/bash
# Smoke-gate canonicalize_GR, then rebuild+compare GR canonical sectors from verified maps.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
{
  echo "=== canonicalize_GR smoke gate ==="
  PYTHONUNBUFFERED=1 $PY reduction/canonicalize_GR.py
  echo
  echo "=== canonical sectors rebuild from verified maps ==="
  PYTHONUNBUFFERED=1 $PY reduction/verify_canonical_sectors_GR.py
} > reduction/logs/gr_canon_verify_v1.log 2>&1
