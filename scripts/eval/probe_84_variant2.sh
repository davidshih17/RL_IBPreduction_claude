#!/bin/bash
# (8,4) probe with variant 2 dedup (resolved_subs-only key) — historical
# variant that empirically cracked (8,4) in 261 steps. Same CLI as
# wide_beam_dedup_v1: bw=40, 16 workers, dedup ON, beam-sort=weight.
# DEDUP_VARIANT=rs env var selects variant 2.
#
# Explicit CLI overrides vs. paper defaults (set 2026-05-28):
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   --beam_width 40               (paper default: 20)
# These override the trianglebox-paper recipe to reproduce the (8,4)
# pentagon-box configuration. NOTE: this script uses onestep_worker.py
# (the Option-F path) — keep here only as a historical probe.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_variant2
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
environment = "PYTHONUNBUFFERED=1 DEDUP_VARIANT=rs"
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
