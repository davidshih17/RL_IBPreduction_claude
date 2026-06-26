#!/bin/bash
# ============================================================================
# CACHE-REUSE EXPERIMENT #2: reduce the highest-weight remaining list_TA target
#   target : TA[1,1,1,1,1,1,1,-1,-3,-2,0]   weight (7,6)
# reusing BOTH prior caches at once via the merged cache dir:
#   cache  : results/cache_85_75_combined   (= (8,5) replay_state + partial (7,5)
#                                              work/results, 68,499 entries)
#            built by reduction/merge_caches.py; --resume-from reads its
#            replay_state.pkl read-only.
#
# Settings = the production v7_fresh round-1 recipe (same basis the caches were
# built in: pentagonbox + --no-paper-masters-only) at 1-CPU serial workers,
# 1000-wide, straggler escalation off, --resume. Own output dir; runs alongside
# the (7,5) orchestrator without conflict (separate work dir + pending set).
# ============================================================================
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_7_6_cache_from_85_75
CACHE=$BASE/results/cache_85_75_combined
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,-1,-3,-2,0"

mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL_STR" \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --resume-from $CACHE \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $OUTDIR/logs/hierarchical.log 2>&1 &
ORCH_PID=$!
set +x
echo "cache-reuse (7,6) orchestrator launched PID=$ORCH_PID"
echo "  target:  $INTEGRAL_STR  (weight 7,6)"
echo "  cache:   $CACHE (--resume-from, 68,499 entries = (8,5)+(7,5))"
echo "  log:     $OUTDIR/logs/hierarchical.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
