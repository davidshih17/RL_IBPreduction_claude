#!/bin/bash
# Smoke-test the in-process onestep_worker_v7.py on ONE round3 target:
# confirm it imports, pins to 8 cores, runs beam_search_v7 at 8/8, and writes an
# orchestrator-format result.pkl (keys: success, original_integral, final_expr,
# path, steps, ...). Output unbuffered to a single log file.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
TARGETS=$BASE/results/pentagonbox_8_5_v6_round3/round3_targets.txt
OUT=$BASE/results/pentagonbox_8_5_v6_round3/smoke
LOG=$OUT/smoke_v7_worker.log
mkdir -p $OUT

TARGET=$(head -1 "$TARGETS")
echo "smoke target: $TARGET" | tee $LOG

PYTHONUNBUFFERED=1 $PY -u $BASE/scripts/eval/onestep_worker_v7.py \
    --topology $TOPO \
    --integral="$TARGET" \
    --output $OUT/smoke_result.pkl \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --device cpu -v --no-paper-masters-only \
  >> $LOG 2>&1 &

echo "smoke PID=$!  log=$LOG  result=$OUT/smoke_result.pkl"
