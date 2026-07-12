#!/bin/bash
# Smoke test of the same-(r,s) symmetry drop detector (SAILIR_SYM_DROP=1):
# rerun one real m2 worker target from the symmetry-rich sector 148 with the
# detector on, then check the emitted rule: any same-(r,s) RHS non-master must be
# router-eliminable (that is the contract the drop relies on).
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
OUT=$BASE/results/smoke_symdrop
mkdir -p $OUT
export SAILIR_SECTOR_RANK=1
export SAILIR_SYM_DROP=1
TARGET='0,0,1,0,2,-1,0,1,-1,0,0'
$PY -u $BASE/reduction/onestep_worker_v7.py \
    --topology $TOPOLOGY --integral="$TARGET" \
    --output $OUT/worker_symdrop.pkl \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 --device cpu -v --v7-cpus 1 2>&1
echo ""
echo "=== contract check: same-(r,s) RHS non-masters must be router-eliminable ==="
$PY $BASE/analysis/check_symdrop_contract.py $OUT/worker_symdrop.pkl "$TARGET" 2>&1
