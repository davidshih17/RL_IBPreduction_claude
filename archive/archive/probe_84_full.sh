#!/bin/bash
# (8,4) reproducer using the FULL (pre-Option-F) code path:
#   scripts/eval/beam_search_full.py
#   scripts/eval/onestep_worker_full.py
#
# apply_action_resolved keeps FULL sols; state.expr contains both target-
# sector and sub-sector content; no IBP replay at worker end; dedup key
# is the full-resolved_subs fingerprint (see beam_search_full.py header).
#
# This is the EMPIRICAL configuration that wide_beam_dedup_v1 (Condor cluster
# 1462640, 2026-05-26T12:48Z) used to crack (8,4) in 261 steps. Bit-identical
# replay verified 2026-05-28 by cluster 1478819 (paths identical=True,
# final_expr identical=True, steps=261, weight (8,4)->(8,3)).
#
# DO NOT switch this script to scripts/eval/beam_search.py (the Option-F path)
# without re-verifying it can still crack (8,4) — empirically it cannot.
#
# Explicit CLI overrides vs. paper defaults (set 2026-05-28):
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   --beam_width 40               (paper default: 20)
# These override the trianglebox-paper recipe to reproduce the wide_beam_
# dedup_v1 configuration that empirically cracked (8,4).
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_full
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker_full.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
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
