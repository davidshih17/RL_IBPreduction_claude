#!/bin/bash
# (8,4) pentagon-box probe with TRIANGLEBOX-PAPER DEFAULTS.
#
# This script intentionally passes NO override flags beyond the topology
# / integral / paths / Condor knobs. All algorithmic settings come from
# the new defaults applied 2026-05-28 in:
#   onestep_worker_full.py  (--beam-sort mixed, --paper-masters-only,
#                            no --dedup-beam-by-content, --beam_width 20)
#
# Code path: scripts/eval/onestep_worker_full.py (FULL, pre-Option-F).
# This is the closest match to the algorithm used by the published
# trianglebox paper.
#
# NOTE: (8,4) was previously cracked only by wide_beam_dedup_v1's
# specific empirical configuration (bw=40, --beam-sort weight,
# --dedup-beam-by-content, --no-paper-masters-only). It is an open
# question whether the paper-default configuration can reduce (8,4)
# within a reasonable step budget — that's exactly what this probe tests.
# probe_84_full.sh remains as the working (8,4) reproducer in case
# this probe fails.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_paperdefaults
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker_full.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --max_steps 5000 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300
environment = "PYTHONUNBUFFERED=1"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 16
request_memory = 32GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
