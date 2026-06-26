#!/bin/bash
# Resume the 3 integrals left unfinished when hexcms (the submit node) rebooted.
# Their orchestrators died with the node, but their partial work + the schedd's
# workers survived on /het + the compute nodes. Rebuild the full combined cache,
# then relaunch the 3 with --resume so each picks up its partial work/results
# (and any results the still-running workers produce) and finishes.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
CACHE=$BASE/results/meta_reduce/_resume_cache

# 1. use the already-built combined cache (built on the first resume attempt)
echo "using existing combined cache: $CACHE"
ls -l $CACHE/replay_state.pkl | awk '{print "  "$5" bytes"}'

# 2. relaunch the 3, resuming their partial work
resume_one() {
  local d=$1 integral=$2
  local OUTDIR=$BASE/results/meta_reduce/$d
  mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
  PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
      --topology $TOPOLOGY --integral="$integral" \
      --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
      --resume-from $CACHE --model-checkpoint $MODEL \
      --beam_width 40 --max_steps 1000000 --prime 1009 \
      --no-paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
      --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
      --check-interval 5 --max-concurrent 1000 --resume \
      > $OUTDIR/logs/resume.log 2>&1 &
  echo "  resumed $d ($integral) PID=$!"
}
# tgt0028 handled separately (its survivor workers are still finishing; will
# re-resume once cluster 1830090 drains, to avoid re-duplicating).
resume_one tgt0474_w5_3 "-2,1,1,1,1,0,0,1,-1,0,0"
resume_one tgt0610_w5_3 "-1,1,1,1,1,0,0,1,-2,0,0"
