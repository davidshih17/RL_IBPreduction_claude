#!/bin/bash
# Login-node smoke for the enumerate fork-pool: run the SAME (7,4) reduction for
# a few steps at --n-workers 1 (serial path) and --n-workers 4 (fork-pool), then
# assert the beams are byte-identical. Catches fork crashes + any divergence
# cheaply before committing the full Condor (7,4). Same SAILIR_* flags as the
# real probe (required for bit-identicality). Single short run = a quick sanity,
# not a multi-event job, so login-node is fine. All output -> one log per run.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
STEPS=6

COMMON_ARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps $STEPS --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --device cpu --model-batch-chunk 8"
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
  SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_P4_FORK=1

run_one () {  # $1 = n_workers, $2 = outdir tag
  local nw=$1 tag=$2
  local OUTDIR=$BASE/results/$tag
  mkdir -p $OUTDIR
  echo "=== run n_workers=$nw -> $OUTDIR ===" | tee $OUTDIR/run.log
  $PYTHON -u $SCRIPT $COMMON_ARGS --n-workers $nw \
    --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl >> $OUTDIR/run.log 2>&1
  echo "=== n_workers=$nw done ===" | tee -a $OUTDIR/run.log
}

run_one 1 smoke_p4_nw1
run_one 4 smoke_p4_nw4

echo ""
echo "=============== EXACT BEAM COMPARE (nw1 vs nw4) ==============="
$PYTHON $BASE/scripts/eval/compare_beams_exact.py \
  $BASE/results/smoke_p4_nw1/result.pkl \
  $BASE/results/smoke_p4_nw4/result.pkl
