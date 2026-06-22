#!/bin/bash
# Profile STRIP=1 (minw stripped) vs STRIP=0 (full raws) back-to-back on the same
# machine, ~40 steps each, under cProfile. Isolates the time cost of raw
# stripping and shows which function eats it. Unbuffered, logs to logs/.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
LOGDIR=$BASE/logs
mkdir -p $LOGDIR
OUT=$BASE/results/profile_strip
mkdir -p $OUT

ARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
--no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 40 \
--max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
--n-threads 1 --device cpu --model-batch-chunk 8"

for STRIP in 1 0; do
  tag=strip${STRIP}
  echo "=== profiling STRIP=$STRIP -> $tag ==="
  PYTHONUNBUFFERED=1 SAILIR_STRIP_RAWS=$STRIP SAILIR_TABU_CAP=0 \
    $PYTHON -m cProfile -o $OUT/${tag}.prof \
    $SCRIPT $ARGS --tabu \
    --output $OUT/${tag}_result.pkl --ckpt $OUT/${tag}_ckpt.pkl --ckpt-every 9999 \
    > $LOGDIR/profile_${tag}.log 2>&1
  echo "  done STRIP=$STRIP"
done
echo "ALL PROFILES DONE"
