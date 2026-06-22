#!/bin/bash
# Smoke v8 on probe 74 (7,4) with the CONSISTENT total-ordering strip
# (is_active total order + raw-strip total order + single full-ordering target).
# Single worker / single thread (login sanity). Goal: does it now REDUCE
# (mw comes down, n_non_masters -> 0) instead of climbing to (9,*) like the
# inconsistent-strip run did. v7 reduces 74 in 81 steps.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
O=$BASE/results/smoke_v8_74_consistent
mkdir -p $O
export PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 \
       SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1
$PYTHON -u $BASE/scripts/eval/beam_search_v8.py \
  --topology $BASE/topology_input/pentagonbox --model $MODEL \
  --integral=0,1,1,1,1,1,1,1,-4,0,0 --ckpt-every 9999 --tabu --no-exprkeyed \
  --iraws-keep-first 50 --beam-width 40 --max-steps 600 --max-actions 900 \
  --beam-sort weight --no-paper-masters-only --prime 1009 \
  --n-threads 1 --n-workers 1 --device cpu --model-batch-chunk 8 \
  --output $O/r.pkl > $O/run.log 2>&1
echo "EXIT=$? (probe 74)" >> $O/run.log
