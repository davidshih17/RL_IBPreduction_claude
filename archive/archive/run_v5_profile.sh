#!/bin/bash
# 10-step v5 run with V5_PROFILE=1 to see per-phase breakdown.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR=results/v5_profile_${STAMP}
mkdir -p $DIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"

V5_PROFILE=1 PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 10 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --ckpt $DIR/ckpt.pkl --ckpt-every 9999 > $DIR/run.log 2>&1
echo "log: $DIR/run.log"
grep -E "PROF|\[v5 step" $DIR/run.log
