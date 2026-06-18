#!/bin/bash
# Runs cache-ON then cache-OFF back-to-back on the SAME condor node, full (7,4).
# Only difference between the two: SAILIR_NO_SECTOR_CACHE. MEMBD per-process.
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SC=$B/scripts/eval/beam_search_v7.py
A="--topology $B/topology_input/pentagonbox --model $B/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
--integral=0,1,1,1,1,1,1,1,-4,0,0 --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
--max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
--n-threads 1 --device cpu --model-batch-chunk 8 --tabu"
echo "RUNNING ON HOST: $(hostname)"
for NC in 0 1; do
  tag=$([ $NC = 0 ] && echo cacheON || echo cacheOFF)
  echo "=== $tag (SAILIR_NO_SECTOR_CACHE=$NC) ==="
  PYTHONUNBUFFERED=1 SAILIR_NO_SECTOR_CACHE=$NC SAILIR_PACKED_RS=1 SAILIR_STRIP_RAWS=1 \
    SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=5 SAILIR_END_OF_STEP_TRIM=1 \
    MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_TABU_CAP=0 \
    $PY -u $SC $A --output $B/results/abc/$tag.pkl --ckpt $B/results/abc/$tag.ckpt --ckpt-every 999 \
    > $B/logs/abc_$tag.log 2>&1
  echo "$tag done: $(grep -oE 't_total=[0-9.]+s' $B/logs/abc_$tag.log | tail -1)"
done
echo "ABC DONE on $(hostname)"
