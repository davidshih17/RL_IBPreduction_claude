#!/bin/bash
# Re-run both dedup-OFF and dedup-ON for 10 steps, dumping the full beam
# (all 20 states' signatures) at each step. Output is two parallel BEAM_DUMP
# files we can diff line by line to isolate WHICH non-best beam slot first
# differs between the runs.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/beam_divergence
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

export PYTHONUNBUFFERED=1

# DEDUP-OFF run
rm -f $OUTDIR/beam_dedup_off.log
BEAM_DUMP_FULL=$OUTDIR/beam_dedup_off.log $PYTHON -u $BASE/scripts/eval/onestep_worker.py \
    --topology $TOPOLOGY --integral="$INTEGRAL_STR" \
    --output $OUTDIR/result_off.pkl \
    --model-checkpoint $MODEL --beam_width 20 --max_steps 10 \
    --prime 1009 --device cpu -v --n_workers 4 \
    --no-dedup-beam-by-content --no-checkpoint \
    > $OUTDIR/probe_off.out 2>&1

# DEDUP-ON run
rm -f $OUTDIR/beam_dedup_on.log
BEAM_DUMP_FULL=$OUTDIR/beam_dedup_on.log $PYTHON -u $BASE/scripts/eval/onestep_worker.py \
    --topology $TOPOLOGY --integral="$INTEGRAL_STR" \
    --output $OUTDIR/result_on.pkl \
    --model-checkpoint $MODEL --beam_width 20 --max_steps 10 \
    --prime 1009 --device cpu -v --n_workers 4 \
    --no-checkpoint \
    > $OUTDIR/probe_on.out 2>&1

echo "DONE"
ls -la $OUTDIR
