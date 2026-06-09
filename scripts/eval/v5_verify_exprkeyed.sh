#!/bin/bash
# Verify exprkeyed-delta is bit-identical to depth-keyed incremental.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR_DEPTH=results/v5_verify_exprkeyed_${STAMP}/depthkeyed
DIR_EXPR=results/v5_verify_exprkeyed_${STAMP}/exprkeyed
mkdir -p $DIR_DEPTH $DIR_EXPR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"
N=10

echo "=== Run 1: depth-keyed incremental (control) ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --ckpt $DIR_DEPTH/ckpt.pkl --ckpt-every 9999 \
    --ckpt-every-step > $DIR_DEPTH/run.log 2>&1
echo "  exit=$?"
tail -3 $DIR_DEPTH/run.log

echo "=== Run 2: exprkeyed-delta ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --ckpt $DIR_EXPR/ckpt.pkl --ckpt-every 9999 \
    --ckpt-every-step > $DIR_EXPR/run.log 2>&1
echo "  exit=$?"
tail -3 $DIR_EXPR/run.log

echo "=== Per-step diff ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/v5_diff_step_ckpts.py \
    --dir-a $DIR_DEPTH --dir-b $DIR_EXPR --steps $N
