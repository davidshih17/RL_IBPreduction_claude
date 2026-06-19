#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
ARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 8 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --n-threads 2 --n-workers 2 --device cpu --model-batch-chunk 8"
export PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1
for strat in maxweight shortest sumweight last900; do
  O=$BASE/results/smoke_as_$strat; mkdir -p $O
  echo "=== $strat ==="
  SAILIR_ACTION_SELECT=$strat $PYTHON -u $SCRIPT $ARGS --output $O/r.pkl --ckpt $O/c.pkl > $O/run.log 2>&1 \
    && echo "  OK ($(grep -oE 'step +[0-9]+\].*cand=[0-9]+' $O/run.log | tail -1 | grep -oE 'step +[0-9]+'))" \
    || { echo "  CRASH:"; tail -8 $O/run.log; }
done
