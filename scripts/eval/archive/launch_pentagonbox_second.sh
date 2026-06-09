#!/bin/bash
# Launch hierarchical reduction for second list_TA target.
# Cluster expected to be relatively idle vs the first run.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_second
mkdir -p $OUTDIR/logs $OUTDIR/work

cd $BASE

export PYTHONUNBUFFERED=1
/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 1,0,1,0,1,1,0,0,0,-1,-1 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_v2/best_model.pt \
    --beam_width 20 \
    --max_steps 1000000 \
    --prime 1009 \
    > $OUTDIR/logs/hierarchical.log 2>&1
