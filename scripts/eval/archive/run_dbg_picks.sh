#!/bin/bash
# Diagnostic: dump per-task picks for baseline-tabu vs rank-cycle on integral 74,
# first 6 steps, to locate the first divergence. Runs both in background, logs.
set -e
CAP=${1:-60}      # bounded-tabu cap (default 60)
NSTEPS=${2:-9}    # number of steps
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL='0,1,1,1,1,1,1,1,-4,0,0'

DBG=$BASE/results/dbg_picks
BL_DIR=$DBG/baseline
RC_DIR=$DBG/btabu_C${CAP}
mkdir -p $BL_DIR $RC_DIR
rm -f $BL_DIR/picks_step*.pkl $RC_DIR/picks_step*.pkl

COMMON="--topology $TOPOLOGY --model $MODEL --integral=$INTEGRAL \
  --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps $NSTEPS --max-actions 900 --beam-sort weight \
  --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
  --model-batch-chunk 8"

# Baseline (acfe668 instrumented), tabu ON
PYTHONUNBUFFERED=1 SAILIR_PICK_DUMP=$BL_DIR \
  $PYTHON -u $BASE/scripts/eval/beam_search_v6_baseline_dbg.py \
  $COMMON --tabu \
  --output $BL_DIR/result.pkl --ckpt $BL_DIR/ckpt.pkl --ckpt-every 1000 \
  > $DBG/baseline.log 2>&1 &
echo "baseline PID $!"

# Current code, bounded-tabu CAP=60
PYTHONUNBUFFERED=1 SAILIR_PICK_DUMP=$RC_DIR SAILIR_TABU_CAP=$CAP \
  MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  $PYTHON -u $BASE/scripts/eval/beam_search_v6.py \
  $COMMON --tabu \
  --output $RC_DIR/result.pkl --ckpt $RC_DIR/ckpt.pkl --ckpt-every 1000 \
  > $DBG/rankcyc.log 2>&1 &
echo "rankcyc PID $!"

wait
echo "both done"
