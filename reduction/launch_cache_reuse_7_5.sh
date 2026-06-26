#!/bin/bash
# ============================================================================
# CACHE-REUSE EXPERIMENT: reduce a (7,5) list_TA integral one weight level
# below (8,5), REUSING the (8,5) reduction cache.
#
#   target   : TA[1,1,1,1,1,1,1,-1,-3,-1,0]   weight (7,5)
#   cache    : results/pentagonbox_8_5_v7_fresh   (--resume-from, read-only)
#
# Settings match the production pentagonbox_8_5_v7_fresh ROUND-1 run -- which is
# also the basis the resumed cache was built in (so cache keys align and the
# reuse measurement is apples-to-apples):
#     --topology pentagonbox  +  --no-paper-masters-only
#     --use-v7-worker --v7-cpus 1 --worker-memory-gb 4   (1-CPU serial workers)
#     --max-concurrent 1000                              (fan out wide)
#     --straggler*-timeout 1000000000  --check-interval 5  --resume
#     --beam_width 40 --max_steps 1000000 --prime 1009
#
# Output dir follows the same structure as the previous orchestrator jobs:
#     OUTDIR/{logs, work/{logs,results}, reduction.pkl}
# ============================================================================
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_7_5_cache_from_85
CACHE=$BASE/results/pentagonbox_8_5_v7_fresh
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,-1,-3,-1,0"

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
echo "cache-reuse (7,5) orchestrator launched PID=$ORCH_PID"
echo "  target:  $INTEGRAL_STR"
echo "  cache:   $CACHE (--resume-from)"
echo "  log:     $OUTDIR/logs/hierarchical.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
