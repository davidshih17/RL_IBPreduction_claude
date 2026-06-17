#!/bin/bash
# Gating experiment: shared-memory process-pool parallelism on the rs-bound
# kernel (apply_resolved_subs), swept over worker counts, on the DEEP memhog
# resolved_subs. Condor (32GB to load the 626MB checkpoint, 16 cpus for the pool).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/bench_parallel_rs
mkdir -p $OUTDIR
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SCRIPT=$BASE/scripts/eval/bench_parallel_apply_rs.py
CKPT=$BASE/results/memhog_v7/ckpt.pkl     # deep rs (~684 subs, large solutions)

# 50k equations, 80 terms each, 15% sub-keys; sweep 1/2/4/8/16 workers
cat > $OUTDIR/bench.sub <<SUBEOF
universe = vanilla
executable = $PY
arguments = -u $SCRIPT $CKPT 1,2,4,8,16 50000 80 0.15
environment = "PYTHONUNBUFFERED=1"
output = $OUTDIR/bench.out
error  = $OUTDIR/bench.out
log    = $OUTDIR/bench.log
request_cpus = 16
request_memory = 40GB
request_disk = 10GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/bench.sub
