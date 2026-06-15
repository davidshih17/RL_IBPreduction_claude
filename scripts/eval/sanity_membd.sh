#!/bin/bash
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
export PYTHONUNBUFFERED=1 SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=4
$PYTHON -u $BASE/scripts/eval/onestep_worker_v6.py \
  --topology $BASE/topology_input/pentagonbox \
  --integral='0,2,0,0,0,1,1,1,0,0,0' \
  --output /tmp/sanity_membd_result.pkl \
  --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
  --beam_width 40 --max_steps 18 --prime 1009 --device cpu -v --no-paper-masters-only \
  > $BASE/results/sanity_membd.log 2>&1
echo "exit $?"
