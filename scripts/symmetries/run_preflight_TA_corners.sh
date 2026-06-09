#!/bin/bash
set -u
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/symmetries
mkdir -p logs
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
PYTHONUNBUFFERED=1 "$PY" -u preflight_TA_corners.py > logs/preflight_TA_corners.log 2>&1
echo "exit=$?  see logs/preflight_TA_corners.log"
