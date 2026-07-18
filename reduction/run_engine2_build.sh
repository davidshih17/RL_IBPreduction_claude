#!/bin/bash
# Build the GENERAL-engine transform stores for both topologies, then gate:
#   gravity3L: equivalence vs the validated GR-specific store
#   pentagonbox: legacy coverage + orbit-structure identity
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
{
  echo "############ BUILD gravity3L (general engine) ############"
  PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine2.py \
      topology_input/gravity3L \
      topology_input/gravity3L/kira_validate/sectormappings/GR \
      results/gravity3L_transforms_v2.pkl 1009
  echo
  echo "############ GATE gravity3L vs validated GR store ############"
  PYTHONUNBUFFERED=1 $PY reduction/gate_engine2.py gravity3L
  echo
  echo "############ BUILD pentagonbox (general engine) ############"
  PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine2.py \
      topology_input/pentagonbox \
      results/kira_reduce_161/sectormappings/TA \
      results/pentagonbox_transforms_v2.pkl 1009
  echo
  echo "############ GATE pentagonbox vs legacy engine ############"
  PYTHONUNBUFFERED=1 $PY reduction/gate_engine2.py pentagonbox
} > reduction/logs/engine2_build_v4.log 2>&1
