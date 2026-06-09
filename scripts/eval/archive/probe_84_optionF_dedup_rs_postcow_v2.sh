#!/bin/bash
# (8,4) probe — Option-F path (sub-sector stripped from sols throughout)
# with resolved_subs-fingerprint dedup, NO tabu list.
#
# Hypothesis: under Option F (apply_action_resolved_target_only strips sols
# to target-sector content at storage time), does the resolved_subs-
# fingerprint dedup that cracked (8,4) on the FULL path (wide_beam_dedup_v1,
# cluster 1462640) STILL crack (8,4)? If yes, the (8,4) crack survives
# without needing sub-sector "passenger" content in the dedup key — which
# would be a much more principled mechanism than the FULL path uses.
#
# Code path: scripts/eval/onestep_worker.py (Option F, target-only stripped).
# Selectors:
#   DEDUP_VARIANT=rs   — resolved_subs fingerprint dedup (variant 2)
#   NO_TABU=1          — disable visited_exprs tabu list (added LATER than
#                        wide_beam_dedup_v1; reproducing its no-tabu regime)
#
# Explicit CLI overrides vs. paper defaults:
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   --beam_width 40               (paper default: 20)
#
# This script does NOT modify probe_84_full.sh or probe_84_variant2.sh —
# those remain as separate reference probes.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_optionF_dedup_rs_postcow_v2
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
environment = "PYTHONUNBUFFERED=1 DEDUP_VARIANT=rs NO_TABU=1"
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
