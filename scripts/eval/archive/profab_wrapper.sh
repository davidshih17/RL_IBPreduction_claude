#!/bin/bash
# Same-node profiled A/B: n_workers=1 then n_workers=8 (both n_threads=8, both
# V5_PROFILE=1, full (7,4)). Decomposes per-phase wall-clock (P1 enum, P2 model,
# P3 apply/sort, P4 materialize, residual) to show exactly what the fork-pool
# parallelizes vs what stays serial. Same node -> totals are directly comparable
# (no node confound). All output -> one log per config.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 V5_PROFILE=1
COMMON="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --n-threads 8 --device cpu --model-batch-chunk 8"

A=$BASE/results/profab_nw1; mkdir -p $A
echo "### CONFIG A n_workers=1 host=$(hostname) start=$(date +%s)"
$PYTHON -u $SCRIPT $COMMON --n-workers 1 --output $A/result.pkl --ckpt $A/ckpt.pkl > $A/probe.out 2>&1
echo "### CONFIG A done=$(date +%s)"

B=$BASE/results/profab_nw8; mkdir -p $B
echo "### CONFIG B n_workers=8 host=$(hostname) start=$(date +%s)"
$PYTHON -u $SCRIPT $COMMON --n-workers 8 --output $B/result.pkl --ckpt $B/ckpt.pkl > $B/probe.out 2>&1
echo "### CONFIG B done=$(date +%s)  host=$(hostname)"
