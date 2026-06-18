#!/bin/bash
# Same-node A/B isolating the P4-fork cost/benefit: nw8+nthr8, SAILIR_P4_FORK
# OFF (A) then ON (B), both profiled + memory-tracked. Reveals P4 phase time
# (serial vs forked+remap), total time, and peak RSS delta on ONE node.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 V5_PROFILE=1 \
  SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=5
COMMON="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8"

A=$BASE/results/p4ab_off; mkdir -p $A
echo "### A P4_FORK=off host=$(hostname)"
SAILIR_P4_FORK=0 $PYTHON -u $SCRIPT $COMMON --output $A/result.pkl --ckpt $A/ckpt.pkl > $A/probe.out 2>&1
echo "### A done"

B=$BASE/results/p4ab_on; mkdir -p $B
echo "### B P4_FORK=on host=$(hostname)"
SAILIR_P4_FORK=1 $PYTHON -u $SCRIPT $COMMON --output $B/result.pkl --ckpt $B/ckpt.pkl > $B/probe.out 2>&1
echo "### B done host=$(hostname)"
