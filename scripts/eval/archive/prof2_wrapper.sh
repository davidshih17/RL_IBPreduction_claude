#!/bin/bash
# Re-profile CURRENT state (Cython substitute_one + sector cache) to pick the
# next Cython target. cProfile self-time, output -> results/prof2/.
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SC=$B/scripts/eval/beam_search_v7.py; O=$B/results/prof2
echo "HOST: $(hostname)"
PYTHONUNBUFFERED=1 SAILIR_PACKED_RS=1 SAILIR_STRIP_RAWS=1 SAILIR_TABU_CAP=0 \
  $PY -u -m cProfile -o $O/cprof.out $SC \
  --topology $B/topology_input/pentagonbox --model $B/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
  --integral=0,1,1,1,1,1,1,1,-4,0,0 --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 50 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
  --n-threads 1 --device cpu --model-batch-chunk 8 --tabu \
  --output $O/p.pkl --ckpt $O/p.ckpt --ckpt-every 999 > $O/run.out 2>&1
$PY $B/scripts/eval/archive/prof_analyze.py $O/cprof.out > $O/analysis.txt 2>&1
echo "PROF2 DONE"
