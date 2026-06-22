#!/bin/bash
# Longrunner probe under exprkeyed (bounded-anchor) path + O(1) memory accounting.
# Integral 1,1,1,0,0,1,3,1,-2,-1,0 — known hard case, depth-keyed baseline:
#   968 steps, 77.6 GB peak, 25.3 h (worker async_16344).
# Goal: does exprkeyed (a) reduce it successfully, (b) at much lower memory,
#       (c) in how many steps.
# Same beam_search_v6.py invocation as the C0 probes, but WITHOUT --no-exprkeyed
# (so use_exprkeyed=True) and with the O(1) smaps accounting on.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v6.py
OUTDIR=$BASE/results/longrunner_v6_exprkeyed
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

cat > "$OUTDIR/probe.sub" <<SUBEOF
universe = vanilla
executable = $PYTHON
arguments = -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='1,1,1,0,0,1,3,1,-2,-1,0' --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 --tabu --iraws-keep-first 50 --beam-width 40 --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu --model-batch-chunk 8
environment = "PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=5"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 64GB
request_disk = 80GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit "$OUTDIR/probe.sub"
