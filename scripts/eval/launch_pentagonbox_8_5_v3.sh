#!/bin/bash
# (8,5) target via 10x_loop_100 model + checkpoint+resume orchestrator.
# Successor to pentagonbox_8_5 (v2 model) and pentagonbox_8_5_v1_buggy_stragglers.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v3
mkdir -p $OUTDIR/logs $OUTDIR/work

cd $BASE

export PYTHONUNBUFFERED=1
/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 1,1,1,1,1,1,1,1,-5,0,0 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
    --beam_width 20 --max_steps 1000000 --prime 1009 \
    --straggler-timeout 3600 \
    --straggler-cpus 8 \
    --checkpoint-interval 50 \
    --checkpoint-time-seconds 300 \
    > $OUTDIR/logs/hierarchical.log 2>&1
