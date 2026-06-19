#!/bin/bash
# Same-node A/B: full config 8 threads/8 workers vs 4 threads/4 workers (kernel
# on, profiled). Answers "do we need 8 cores or is 4 nearly as good?" and shows
# how P2 model (threads) and P1 enumerate (workers) scale.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 V5_PROFILE=1
BASEARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --device cpu --model-batch-chunk 8"
A=$BASE/results/scale_8x8; mkdir -p $A
echo "### A 8thr/8wk host=$(hostname)"
$PYTHON -u $SCRIPT $BASEARGS --n-threads 8 --n-workers 8 --output $A/result.pkl --ckpt $A/ckpt.pkl > $A/probe.out 2>&1
echo "### A done"
B=$BASE/results/scale_4x4; mkdir -p $B
echo "### B 4thr/4wk host=$(hostname)"
$PYTHON -u $SCRIPT $BASEARGS --n-threads 4 --n-workers 4 --output $B/result.pkl --ckpt $B/ckpt.pkl > $B/probe.out 2>&1
echo "### B done host=$(hostname)"
