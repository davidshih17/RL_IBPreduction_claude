#!/bin/bash
# First GR one-step worker smoke test: easiest benchmark integral
# [1,1,1,1,0,1,1,...] (sector 111, L=6, canonical; FIRE says scaleless-zero,
# so the worker should drain the bucket / reduce it away entirely).
# Login-node single-worker sanity check (1-event equivalent).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
mkdir -p results/gr_worker_smoke
PYTHONUNBUFFERED=1 $PY reduction/onestep_worker_v7.py \
    --topology topology_input/gravity3L \
    --integral '1,1,1,1,0,1,1,0,0,0,0,0,0,0,0' \
    --output results/gr_worker_smoke/smoke1.pkl \
    --model-checkpoint checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --max_steps 300 \
    > reduction/logs/gr_worker_smoke_v1.log 2>&1
