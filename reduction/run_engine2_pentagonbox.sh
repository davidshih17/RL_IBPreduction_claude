#!/bin/bash
# Pentagonbox instance of the general engine + gate vs the legacy engine.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
{
  echo "############ BUILD pentagonbox (general engine) ############"
  PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine2.py \
      topology_input/pentagonbox \
      results/kira_reduce_161/sectormappings/TA \
      results/pentagonbox_transforms_v2.pkl 1009
  echo
  echo "############ GATE pentagonbox vs legacy engine ############"
  PYTHONUNBUFFERED=1 $PY reduction/gate_engine2.py pentagonbox
} > reduction/logs/engine2_pentagonbox_v3.log 2>&1
