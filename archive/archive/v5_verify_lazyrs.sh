#!/bin/bash
# Verify LAZY_RS:
#  1. Bit-identical to no-lazy via per-step thick ckpt diff (8 steps).
#  2. Late-step timing speedup: resume from saved late ckpt, run a few more steps
#     with and without lazy_rs, compare wall.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR=results/v5_verify_lazyrs_${STAMP}
mkdir -p $DIR/eager $DIR/lazy $DIR/eager_late $DIR/lazy_late
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"

echo "=== STEP 1: bit-identical verification (8 steps from scratch) ==="
echo "  -- eager (no lazy_rs) --"
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 8 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 --no-lazy-rs \
    --ckpt $DIR/eager/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/eager/run.log 2>&1
echo "    exit=$?  $(grep 't_total' $DIR/eager/run.log | tail -1)"

echo "  -- lazy --"
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 8 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --ckpt $DIR/lazy/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/lazy/run.log 2>&1
echo "    exit=$?  $(grep 't_total' $DIR/lazy/run.log | tail -1)"

echo "  -- diff --"
$PYTHON -u scripts/eval/v5_diff_step_ckpts.py \
    --dir-a $DIR/eager --dir-b $DIR/lazy --steps 8

echo ""
echo "=== STEP 2: late-step timing (resume from probe_84 step-250 ckpt, run +4 steps) ==="
echo "  -- eager (no lazy_rs) --"
V5_PROFILE=1 PYTHONUNBUFFERED=1 timeout 900 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 254 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 --no-lazy-rs \
    --resume-from /tmp/probe84_step250_ckpt.pkl \
    --ckpt $DIR/eager_late/ckpt.pkl > $DIR/eager_late/run.log 2>&1
echo "    exit=$?"
grep -E "PROF|cts:|\[v5 step" $DIR/eager_late/run.log

echo ""
echo "  -- lazy --"
V5_PROFILE=1 PYTHONUNBUFFERED=1 timeout 900 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 254 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --resume-from /tmp/probe84_step250_ckpt.pkl \
    --ckpt $DIR/lazy_late/ckpt.pkl > $DIR/lazy_late/run.log 2>&1
echo "    exit=$?"
grep -E "PROF|cts:|\[v5 step" $DIR/lazy_late/run.log
