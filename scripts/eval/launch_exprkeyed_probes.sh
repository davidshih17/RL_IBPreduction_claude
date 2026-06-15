#!/bin/bash
# Launch exprkeyed (bounded-anchor) variants of the canonical v6 C0 probes.
# IDENTICAL to probe_{84,74}_v6_btabu_C0.sh EXCEPT --no-exprkeyed is DROPPED,
# so use_exprkeyed=True (beam_search_v6.py default) -> the bounded
# compute_indirect_substituted_exprkeyed_delta path (the memory improvement).
# Goal: verify bit-identical results vs the stored baselines:
#   (8,4) path_len=182 ; (7,4) path_len=81 ; memhog path_len=720
# plus measure the memhog's new peak memory (was 20.3GB depth-keyed).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v6.py

# name | integral | max_steps | extra_env
submit_probe () {
  local NAME="$1" INTEGRAL="$2" MAXSTEPS="$3" EXTRAENV="$4"
  local OUTDIR=$BASE/results/$NAME
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  cat > "$OUTDIR/probe.sub" <<SUBEOF
universe = vanilla
executable = $PYTHON
arguments = -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='$INTEGRAL' --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 --tabu --iraws-keep-first 50 --beam-width 40 --max-steps $MAXSTEPS --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu --model-batch-chunk 8
environment = "PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 $EXTRAENV"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 32GB
request_disk = 50GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
  condor_submit "$OUTDIR/probe.sub"
}

# Fast bit-identical checks (compare path_len/masters to baseline 182 / 81):
submit_probe probe_84_v6_exprkeyed "-1,2,1,0,1,2,1,1,-3,0,0" 5000 ""
submit_probe probe_74_v6_exprkeyed "0,1,1,1,1,1,1,1,-4,0,0"  5000 ""
# Memhog memory-win check (was 20.3GB / path_len 720 depth-keyed), with O(1) accounting:
submit_probe memhog_v6_exprkeyed   "1,1,1,0,-2,1,1,1,0,0,0"  1000 "SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=5"
