#!/bin/bash
# Round 2 of the (8,5) hierarchical reduction: re-reduce ONLY the 361 tabu-
# trapped integrals from round 1 (which now succeed via the aggressive-tabu +
# exhaustion-cycle worker), pulling round-1's full cache READ-ONLY and writing
# all new results to a fresh parallel directory. The 769 long-running/OOM
# integrals are NOT reachable from the 361 seed expr, so they are set aside.
#
# Reuses hierarchical_reduction.py unchanged except for the new --resume-from
# and --reduce-only flags.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
ROUND1=$BASE/results/pentagonbox_8_5_v6
OUTDIR=$BASE/results/pentagonbox_8_5_v6_round2
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"      # original start (only used for config parity)
TARGETS=$ROUND1/round2_targets_361.txt

DRYRUN=""
[ "$1" = "--dry-run" ] && DRYRUN="--dry-run"

PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --resume-from $ROUND1 \
    --reduce-only $TARGETS \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v6-worker \
    --worker-memory-gb 16 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 10000 \
    $DRYRUN \
  > $OUTDIR/logs/hierarchical_round2.log 2>&1 &

ORCH_PID=$!
echo "Round-2 orchestrator launched PID=$ORCH_PID ${DRYRUN}"
echo "  log:     $OUTDIR/logs/hierarchical_round2.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
