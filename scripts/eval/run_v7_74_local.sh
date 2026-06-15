#!/bin/bash
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
OUTDIR=$BASE/results/probe_74_v7
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
cd "$BASE/scripts/eval"
PYTHONUNBUFFERED=1 $PY beam_search_v7.py --topology "$BASE/topology_input/pentagonbox" \
  --model "$MODEL" --integral='0,1,1,1,1,1,1,1,-4,0,0' \
  --output "$OUTDIR/result.pkl" --ckpt "$OUTDIR/ckpt.pkl" --ckpt-every 50 \
  --tabu --iraws-keep-first 50 --beam-width 40 --max-steps 5000 --max-actions 900 \
  --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 \
  --device cpu --model-batch-chunk 8 > "$OUTDIR/run.log" 2>&1
echo "EXIT=$?" >> "$OUTDIR/run.log"
