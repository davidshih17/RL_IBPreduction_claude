#!/bin/bash
# Two profiles of the single-thread PACKED v7 path (what Option-B/Cython speeds up):
#  (1) V5_PROFILE -> accurate per-phase wall-clock (model vs enumerate/aux vs apply)
#  (2) cProfile   -> function-level self-time (which loops/kernels to Cythonize)
# Single-thread, no apply workers -> clean baseline. ~50 steps of the (7,4) problem.
set -e
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$B/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$B/topology_input/pentagonbox
SCRIPT=$B/scripts/eval/beam_search_v7.py
OUT=$B/results/prof_packed; mkdir -p $OUT; mkdir -p $B/logs
ARGS="--topology $TOPO --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
--no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 50 \
--max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
--n-threads 1 --device cpu --model-batch-chunk 8 --tabu"

echo "=== (1) V5_PROFILE phase breakdown ==="
PYTHONUNBUFFERED=1 V5_PROFILE=1 SAILIR_PACKED_RS=1 SAILIR_APPLY_WORKERS=1 SAILIR_TABU_CAP=0 \
  $PY -u $SCRIPT $ARGS --output $OUT/p1.pkl --ckpt $OUT/p1_ckpt.pkl --ckpt-every 999 \
  > $B/logs/prof_phase.log 2>&1
echo "phase run done"

echo "=== (2) cProfile function self-time ==="
PYTHONUNBUFFERED=1 SAILIR_PACKED_RS=1 SAILIR_APPLY_WORKERS=1 SAILIR_TABU_CAP=0 \
  $PY -u -m cProfile -o $OUT/cprof.out $SCRIPT $ARGS \
  --output $OUT/p2.pkl --ckpt $OUT/p2_ckpt.pkl --ckpt-every 999 \
  > $B/logs/prof_cfn.log 2>&1
echo "cprofile run done"

$PY $B/scripts/eval/archive/prof_analyze.py $OUT/cprof.out >> $B/logs/prof_cfn.log 2>&1
echo "ALL PROF DONE"
