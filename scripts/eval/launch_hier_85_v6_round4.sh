#!/bin/bash
# Round 4 of the (8,5) hierarchical reduction. Targets = the 3 non-masters that
# round1+2+3 left unreduced (the orchestrator's old unit-bag seed had cancelled
# them to 0 and killed their jobs; the seed bug is now fixed). Resumes from the
# COMBINED round1+2+3 cache (results/.../round4/replay_state.pkl, 103,621 entries)
# and replays the REAL (8,5) start through it (true coefficients), so these 3
# appear with their real weights (558/598/50) and get reduced.
#
# With the bug fix, the orchestrator's per-iteration "X masters, Y non-masters"
# line is now an honest live status of the real start reduction.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
R4=$BASE/results/pentagonbox_8_5_v6_round4
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"
TARGETS=$R4/round4_targets.txt
mkdir -p $R4/logs $R4/work/logs $R4/work/results

if [ ! -f "$TARGETS" ]; then echo "ERROR: $TARGETS missing" >&2; exit 1; fi
echo "round4 targets: $(wc -l < $TARGETS)"

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $R4/reduction.pkl \
    --work-dir $R4/work \
    --resume-from $R4 \
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
  > $R4/logs/hierarchical_round4.log 2>&1 &
ORCH_PID=$!
set +x
echo "Round-4 orchestrator launched PID=$ORCH_PID (REAL-start seed, fixed)"
echo "  log:    $R4/logs/hierarchical_round4.log"
echo "  result: $R4/reduction.pkl"
