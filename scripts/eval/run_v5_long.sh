#!/bin/bash
# v5 long-run: 500 steps max, ckpt every 50, on (8,4) starting integral.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
RUNDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/v5_long_${STAMP}
mkdir -p $RUNDIR
LOG=$RUNDIR/run.log
OUT=$RUNDIR/result.pkl
CKPT=$RUNDIR/ckpt.pkl
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt

PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO \
    --model $MODEL \
    --integral='-1,2,1,0,1,2,1,1,-3,0,0' \
    --beam-width 40 \
    --max-steps 500 \
    --max-actions 900 \
    --beam-sort mixed \
    --no-paper-masters-only \
    --prime 1009 \
    --n-threads 1 \
    --device cpu \
    --output $OUT \
    --ckpt $CKPT \
    --ckpt-every 50 > $LOG 2>&1
echo "LOG=$LOG"
echo "OUT=$OUT"
echo "CKPT=$CKPT"
tail -20 $LOG
