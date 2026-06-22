#!/bin/bash
# Quick (8,4) memory-probe to measure where 40 MB/step actually goes.
# Runs 50 steps, then dumps per-survivor cu length, key count, and deep_size.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_memprobe
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/probe_84_delta_memprobe.py --max-steps 50
environment = "PYTHONUNBUFFERED=1 DELTA_REBUILD_INTERVAL=0"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 16GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
