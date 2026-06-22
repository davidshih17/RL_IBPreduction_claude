#!/bin/bash
# REVERSE order: NEW (SEED_MERGE=1) first, then OLD (=0), SAME node. If NEW is
# still slower running FIRST -> real code effect. If OLD (second) is slower ->
# order/environment. Output -> results/abseed_rev/.
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SC=$B/scripts/eval/beam_search_v7.py; O=$B/results/abseed_rev
A="--topology $B/topology_input/pentagonbox --model $B/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
--integral=0,1,1,1,1,1,1,1,-4,0,0 --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
--max-steps 60 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
--n-threads 1 --device cpu --model-batch-chunk 8 --tabu"
echo "HOST: $(hostname)"
for SM in 1 0; do
  tag=$([ $SM = 0 ] && echo OLDdelta || echo NEWdelta)
  PYTHONUNBUFFERED=1 SAILIR_SEED_MERGE=$SM SAILIR_PACKED_RS=1 SAILIR_STRIP_RAWS=1 SAILIR_TABU_CAP=0 \
    $PY -u $SC $A --output $O/$tag.pkl --ckpt $O/$tag.ckpt --ckpt-every 999 > $O/$tag.out 2>&1
  echo "$tag done: $(grep -oE 't_total=[0-9.]+s' $O/$tag.out | tail -1)"
done
echo "ABSEEDREV DONE on $(hostname)"
