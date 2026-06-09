#!/bin/bash
# Time fresh vs incremental over 50 steps to see crossover.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR=results/v5_bench_incr_${STAMP}
mkdir -p $DIR/fresh $DIR/incr
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"
N=50

echo "=== fresh (no incremental) ==="
time PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-incremental-aux \
    --ckpt $DIR/fresh/ckpt.pkl --ckpt-every 9999 > $DIR/fresh/run.log 2>&1
echo "fresh total: $(grep '^\[v5 step' $DIR/fresh/run.log | tail -1 | grep -oP 't_total=[0-9.]+s')"
echo ""
echo "=== incremental ==="
time PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps $N --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu \
    --ckpt $DIR/incr/ckpt.pkl --ckpt-every 9999 > $DIR/incr/run.log 2>&1
echo "incr total: $(grep '^\[v5 step' $DIR/incr/run.log | tail -1 | grep -oP 't_total=[0-9.]+s')"
