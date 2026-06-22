#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_v6_lru
mkdir -p $OUTDIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v6.py

cat > $OUTDIR/probe.sub <<JDL
universe = vanilla
executable = $PYTHON
arguments = -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='-1,2,1,0,1,2,1,1,-3,0,0' --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu
environment = "PYTHONUNBUFFERED=1 SAILIR_RAW_EQ_CACHE_CAP=50000"
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
JDL
condor_submit $OUTDIR/probe.sub
