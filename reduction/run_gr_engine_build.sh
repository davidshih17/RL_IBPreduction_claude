#!/bin/bash
# Build + gate the GR symmetry transform store (symmetry_engine_GR.py).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine_GR.py > reduction/logs/gr_engine_build_v3.log 2>&1
