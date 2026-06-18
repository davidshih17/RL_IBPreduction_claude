#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
OUT=$BASE/results/regprobe; mkdir -p $OUT
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_REG_PROBE=1
$PYTHON -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --output $OUT/result.pkl --ckpt $OUT/ckpt.pkl --ckpt-every 50 --tabu --no-exprkeyed \
  --iraws-keep-first 50 --beam-width 40 --max-steps 45 --max-actions 900 --beam-sort weight \
  --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 1 --device cpu \
  --model-batch-chunk 8 > $OUT/run.log 2>&1
echo "DONE"
