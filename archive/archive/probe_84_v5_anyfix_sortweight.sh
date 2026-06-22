#!/bin/bash
# (8,4) probe with the any() termination fix AND --beam-sort weight (not mixed).
# Other args match probe_84_v5_lazyrs: LAZY_RS + iraws-keep-first 50 + tabu.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_v5_anyfix_sortweight
mkdir -p $OUTDIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/beam_search_v5.py --topology $TOPOLOGY --model $MODEL --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 5000 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu
environment = "PYTHONUNBUFFERED=1"
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
EOF

condor_submit $OUTDIR/probe.sub
