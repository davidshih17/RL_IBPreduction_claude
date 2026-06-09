#!/bin/bash
# Probe: can the model handle a high-r low-s L7 integral as easily as the
# symmetric ISP-heavy one? Target: I[0,5,1,1,1,1,1,1,0,0,0] weight (11,0).
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_probe_l7_high_r
mkdir -p $OUTDIR/logs $OUTDIR/work

cd $BASE

export PYTHONUNBUFFERED=1
/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 0,5,1,1,1,1,1,1,0,0,0 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_v2/best_model.pt \
    --beam_width 20 \
    --max_steps 1000000 \
    --prime 1009 \
    > $OUTDIR/logs/hierarchical.log 2>&1
