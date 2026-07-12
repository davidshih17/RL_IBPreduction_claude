#!/bin/bash
# Single-worker smoke run of the sector-senior order (SAILIR_SECTOR_RANK=1):
# one onestep_worker_v7 on the gate_small target, then verify the emitted rule
# descends strictly in the NEW order (login-node 1-worker sanity check).
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
OUT=$BASE/results/smoke_sectorrank
mkdir -p $OUT
export SAILIR_SECTOR_RANK=1
$PY -u $BASE/reduction/onestep_worker_v7.py \
    --topology $TOPOLOGY --integral='1,-1,1,0,1,1,0,0,0,0,0' \
    --output $OUT/gate_small_worker.pkl \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 --device cpu -v --v7-cpus 1 2>&1
echo ""
echo "=== rule-descent check under the NEW order ==="
$PY $BASE/analysis/check_worker_rule_descent.py $OUT/gate_small_worker.pkl '1,-1,1,0,1,1,0,0,0,0,0' 2>&1
