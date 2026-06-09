#!/bin/bash
# (7,4) probe — Option-F path (sub-sector stripped from sols throughout)
# with DEDUP_VARIANT=expr_rs: dedup key is the PAIR
#   (target-sector expr,  target-sector resolved_subs fingerprint)
#
# Companion to probe_84_optionF_dedup_expr_rs.sh. Tests whether the
# expr_rs dedup that cracks (8,4) bit-identically to wide_beam_dedup_v1
# (cluster 1478829 confirmed 2026-05-28) ALSO succeeds on (7,4) — or
# whether dedup-on hurts (7,4) and we need a beam-diversity mechanism
# (e.g. per-parent quota) to restore it.
#
# Reference: probe_74_paperdefaults (cluster 1478824, mixed beam,
# paper-masters-only, dedup OFF) cracked (7,4) at step 360 in 3138s.
# This probe uses the (8,4)-style overrides (weight sort, dedup ON,
# no paper-masters, bw=40); we want to know if those settings
# preserve (7,4) success too.
#
# Code path: scripts/eval/onestep_worker.py (Option F, target-only stripped).
# Selectors:
#   DEDUP_VARIANT=expr_rs   — strict 2-key dedup (expr + rs, both target sector)
#   NO_TABU=1               — disable visited_exprs tabu list
#
# Explicit CLI overrides vs. paper defaults:
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   --beam_width 40               (paper default: 20)
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_optionF_dedup_expr_rs
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
environment = "PYTHONUNBUFFERED=1 DEDUP_VARIANT=expr_rs NO_TABU=1"
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
