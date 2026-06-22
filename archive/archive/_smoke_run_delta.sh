#!/bin/bash
# Quick functional run: 10 steps on (8,4) integral. Login node, ~1 min.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

OUTDIR=$BASE/results/delta_smoke
mkdir -p $OUTDIR
LOG=$OUTDIR/run.log

cd $BASE
PYTHONUNBUFFERED=1 $PYTHON scripts/eval/delta_onestep_worker.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL_STR" \
    --output $OUTDIR/result.pkl \
    --model-checkpoint $MODEL \
    --beam_width 40 \
    --max_steps 10 \
    --prime 1009 \
    --device cpu \
    -v \
    --no-paper-masters-only \
    > $LOG 2>&1

echo "Exit: $?"
echo "Log tail:"
tail -40 $LOG
