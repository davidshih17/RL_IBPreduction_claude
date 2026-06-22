#!/bin/bash
# Scan the deepest memhog checkpoint for per-position integral power ranges.
# Submitted to Condor (the checkpoint unpickles to several GB).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/scan_integral_ranges
mkdir -p $OUTDIR
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SCRIPT=$BASE/scripts/eval/scan_integral_ranges.py
CKPT=$BASE/results/memhog_v7/ckpt.pkl     # deepest (626MB, step ~700)

cat > $OUTDIR/scan.sub <<SUBEOF
universe = vanilla
executable = $PY
arguments = -u $SCRIPT $CKPT
environment = "PYTHONUNBUFFERED=1"
output = $OUTDIR/scan.out
error  = $OUTDIR/scan.out
log    = $OUTDIR/scan.log
request_cpus = 1
request_memory = 32GB
request_disk = 10GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/scan.sub
