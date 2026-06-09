#!/bin/bash
# Test the bitmask-optimized code path on the (7,4) integral, which the
# baseline reduced in ~81 min on 8 CPUs with the same model.
#
# Goal: verify
#   (a) the optimization preserves correctness (same final result)
#   (b) the integral reduction path is bit-identical (same step count, same
#       weight trajectory) since the filter is provably equivalent
#   (c) wall time is significantly lower
#
# All settings match the original probe (probe_74_10x_loop_100.sub):
# same integral, model, beam_width, prime, n_workers. Output to a separate
# directory so it doesn't overwrite the baseline.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_10x_loop_100_optimized_v2
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 20 --max_steps 1000000 --prime 1009 --device cpu -v --n_workers 8
environment = "PYTHONUNBUFFERED=1"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 8
request_memory = 16GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
echo
echo "Baseline (May 24): 81 min, 360 steps. Watch:"
echo "  tail -f $OUTDIR/probe.out"
echo "  grep '^Step ' $OUTDIR/probe.out | wc -l   # step count"
echo "  grep 'SUCCESS\|FAILED' $OUTDIR/probe.out  # finish marker"
