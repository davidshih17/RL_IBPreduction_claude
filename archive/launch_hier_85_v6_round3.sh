#!/bin/bash
# Round 3 of the (8,5) hierarchical reduction, using the CURRENT beam_search_v7
# via onestep_worker_v7.py (--use-v7-worker) at --n-threads 8 --n-workers 8 with
# ALL v7 settings (SUCCESS_TOTAL total-weight single-step success, (r,s) maxweight
# action-cap, GNU MKL layer + affinity pin so the 8-cpu fork pool stays <= 8 on
# Condor).
#
# Targets = the REMAINING non-masters from the replay of round1 + round2
# (built by make_round3.py -> round3_targets.txt). The combined round1+round2
# cache is loaded READ-ONLY from round3/replay_state.pkl via --resume-from.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
ROUND3=$BASE/results/pentagonbox_8_5_v6_round3
OUTDIR=$ROUND3
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"        # the round-1 start (config parity)
TARGETS=$ROUND3/round3_targets.txt

if [ ! -f "$TARGETS" ]; then
  echo "ERROR: $TARGETS not found — run make_round3.py first." >&2
  exit 1
fi
echo "round3 targets: $(wc -l < $TARGETS)"

# NOTE: --dry-run is deliberately NOT plumbed here. This launcher ALWAYS submits
# for real. (A previous stray --dry-run silently produced a no-op run.)

set -x   # echo the exact python argv into the log so it is auditable
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --resume-from $ROUND3 \
    --reduce-only $TARGETS \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker \
    --worker-memory-gb 48 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 100 \
  > $OUTDIR/logs/hierarchical_round3.log 2>&1 &

ORCH_PID=$!
set +x
echo "Round-3 orchestrator launched PID=$ORCH_PID (REAL submit, no dry-run)"
echo "  workers: onestep_worker_v7.py (beam_search_v7 @ 8/8, all v7 settings)"
echo "  log:     $OUTDIR/logs/hierarchical_round3.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
