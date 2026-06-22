#!/bin/bash
# One-shot probe that dumps the FIRST duplicate pair found by the dedup so we
# can run the model on each state and compare outputs.
#
# Runs login-node (fast, ~30 steps) — does not need Condor.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_dump_dup_pair
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

export PYTHONUNBUFFERED=1
export DEDUP_DEBUG=1
export DEDUP_DUMP_PAIR=$OUTDIR/dup_pair.pkl

$PYTHON -u $BASE/scripts/eval/onestep_worker.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL_STR" \
    --output $OUTDIR/result.pkl \
    --model-checkpoint $MODEL \
    --beam_width 20 \
    --max_steps 5 \
    --prime 1009 \
    --device cpu \
    -v \
    --n_workers 4 \
    --no-checkpoint \
    > $OUTDIR/probe.out 2>&1
