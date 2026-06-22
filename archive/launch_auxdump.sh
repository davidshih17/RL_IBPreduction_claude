#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v6.py
submit () {
  local NAME="$1" EXPRFLAG="$2"
  local OUTDIR=$BASE/results/$NAME
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  cat > "$OUTDIR/probe.sub" <<SUBEOF
universe = vanilla
executable = $PYTHON
arguments = -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='0,1,1,1,1,1,1,1,-4,0,0' --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 1000 --tabu $EXPRFLAG --iraws-keep-first 50 --beam-width 40 --max-steps 30 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu --model-batch-chunk 8
environment = "PYTHONUNBUFFERED=1 SAILIR_DUMP_AUX_STEPS=10,25"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 8GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
  condor_submit "$OUTDIR/probe.sub"
}
submit auxdump_74_depthkeyed "--no-exprkeyed"
submit auxdump_74_exprkeyed   ""
