#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
ARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 6 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --device cpu --model-batch-chunk 8 --n-workers 1"
export PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1
F=$BASE/results/smoke_pa_fast; mkdir -p $F
$PYTHON -u $SCRIPT $ARGS --output $F/result.pkl --ckpt $F/ckpt.pkl > $F/run.log 2>&1
L=$BASE/results/smoke_pa_legacy; mkdir -p $L
SAILIR_PHASEA_LEGACY=1 $PYTHON -u $SCRIPT $ARGS --output $L/result.pkl --ckpt $L/ckpt.pkl > $L/run.log 2>&1
echo "=== EXACT BEAM COMPARE (fast vs legacy) ==="
$PYTHON $BASE/scripts/eval/compare_beams_exact.py $F/result.pkl $L/result.pkl
