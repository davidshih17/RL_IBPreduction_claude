#!/bin/bash
# Run v5 twice (incremental aux ON vs OFF), with --ckpt-every-step,
# then diff every step file to confirm bit-identical state.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR_FRESH=results/v5_verify_incr_${STAMP}/fresh
DIR_INCR=results/v5_verify_incr_${STAMP}/incr
mkdir -p $DIR_FRESH $DIR_INCR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"
N_STEPS=10

echo "=== Run 1: NO incremental (fresh aux every step) ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N_STEPS --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --no-incremental-aux --ckpt $DIR_FRESH/ckpt.pkl --ckpt-every 9999 \
    --ckpt-every-step > $DIR_FRESH/run.log 2>&1
echo "  fresh exit=$?"
tail -3 $DIR_FRESH/run.log

echo "=== Run 2: WITH incremental ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N_STEPS --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --ckpt $DIR_INCR/ckpt.pkl --ckpt-every 9999 \
    --ckpt-every-step > $DIR_INCR/run.log 2>&1
echo "  incr exit=$?"
tail -3 $DIR_INCR/run.log

echo "=== Per-step diff ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/v5_diff_step_ckpts.py \
    --dir-a $DIR_FRESH --dir-b $DIR_INCR --steps $N_STEPS
