#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/profab
mkdir -p $OUTDIR
WRAP=$BASE/scripts/eval/archive/profab_wrapper.sh
chmod +x $WRAP
cat > $OUTDIR/probe.sub <<SUBEOF
universe = vanilla
executable = $WRAP
output = $OUTDIR/wrap.out
error  = $OUTDIR/wrap.err
log    = $OUTDIR/wrap.log
request_cpus = 8
request_memory = 32GB
request_disk = 50GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/probe.sub
