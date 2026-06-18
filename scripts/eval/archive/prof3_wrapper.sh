#!/bin/bash
# Re-profile CURRENT clean committed state (sector cache + substitute_one +
# contains_sorted). (1) V5_PROFILE phase breakdown (parallelizable vs serial),
# (2) cProfile self-time (which function is the top target). -> results/prof3/.
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SC=$B/scripts/eval/beam_search_v7.py; O=$B/results/prof3
A="--topology $B/topology_input/pentagonbox --model $B/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
--integral=0,1,1,1,1,1,1,1,-4,0,0 --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
--max-steps 50 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 \
--n-threads 1 --device cpu --model-batch-chunk 8 --tabu"
echo "HOST: $(hostname)"
echo "=== (1) V5_PROFILE phase breakdown ==="
PYTHONUNBUFFERED=1 V5_PROFILE=1 SAILIR_PACKED_RS=1 SAILIR_STRIP_RAWS=1 SAILIR_TABU_CAP=0 \
  $PY -u $SC $A --output $O/phase.pkl --ckpt $O/phase.ckpt --ckpt-every 999 > $O/phase.out 2>&1
echo "=== (2) cProfile ==="
PYTHONUNBUFFERED=1 SAILIR_PACKED_RS=1 SAILIR_STRIP_RAWS=1 SAILIR_TABU_CAP=0 \
  $PY -u -m cProfile -o $O/cprof.out $SC $A --output $O/c.pkl --ckpt $O/c.ckpt --ckpt-every 999 > $O/crun.out 2>&1
$PY $B/scripts/eval/archive/prof_analyze.py $O/cprof.out > $O/analysis.txt 2>&1
echo "PROF3 DONE"
