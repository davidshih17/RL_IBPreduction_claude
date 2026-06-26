#!/bin/bash
# ============================================================================
# Relaunch tgt0028 (the ONLY unfinished list_TA target after the power outage)
# via the FAST recovery path:
#   - REUSE the 5557 intact completed one-step reductions in work/results
#     (--resume; verified uncorrupted by check_tgt0028_pkls.py — the outage only
#      killed the live orchestrator + the one in-flight worker, not finished pkls)
#   - REUSE the shared cascade cache (--resume-from _burst_cache, 50 MB)
#   - run the remaining HARD tail integrals (e.g. 0,0,4,0,0,1,-1,1,0,-1,0) with
#     16-cpu v7 workers (--v7-cpus 16 -> 18 cpus / 12 GB each) so they finish
#     much faster than the 1-cpu grind that was the original bottleneck.
# The orphan worker 1830090.0 (0,0,4,...) was condor_rm'd first, so there is no
# duplicate 0,0,4,... worker when the orchestrator resubmits it at 16 cpus.
#
# Target: TA[1,1,1,1,1,1,-1,1,-2,0,-1]  weight (7,4)
# New log (does NOT overwrite the original): logs/hierarchical_resume16.log
# ============================================================================
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
RUNDIR=$BASE/results/meta_reduce/tgt0028_w7_4
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
CACHE=$BASE/results/meta_reduce/_burst_cache
INTEGRAL="1,1,1,1,1,1,-1,1,-2,0,-1"
LOG=$RUNDIR/logs/hierarchical_postglitch.log

mkdir -p $RUNDIR/logs $RUNDIR/work/logs $RUNDIR/work/results

set -x
nohup env PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL" \
    --output $RUNDIR/reduction.pkl \
    --work-dir $RUNDIR/work \
    --resume-from $CACHE \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker \
    --v7-cpus 16 \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $LOG 2>&1 &
ORCH_PID=$!
set +x
disown
echo "tgt0028 16-cpu resume orchestrator launched PID=$ORCH_PID"
echo "  integral: $INTEGRAL  (weight 7,4)"
echo "  log:      $LOG"
echo "  reuse:    $RUNDIR/work/results (5557 intact pkls) + --resume-from $CACHE"
