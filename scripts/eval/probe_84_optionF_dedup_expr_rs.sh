#!/bin/bash
# (8,4) probe — Option-F path (sub-sector stripped from sols throughout)
# with DEDUP_VARIANT=expr_rs: dedup key is the PAIR
#   (target-sector expr,  target-sector resolved_subs fingerprint)
#
# Only change vs. probe_84_optionF_dedup_rs.sh: DEDUP_VARIANT=rs → expr_rs.
# Under Option F both halves of the key are target-sector content only
# (no sub-sector "passenger" data anywhere). This tests whether adding
# the expr to the dedup key preserves the (8,4) crack — the 'rs' variant
# already cracked it bit-identically at step 261. If 'expr_rs' also
# cracks (likely the same or fewer beam states pass dedup, since the key
# is stricter), it's a more principled / interpretable dedup criterion:
# two states are duplicates iff they have the same current expression
# AND the same substitution history.
#
# Code path: scripts/eval/onestep_worker.py (Option F, target-only stripped).
# Selectors:
#   DEDUP_VARIANT=expr_rs   — strict 2-key dedup
#   NO_TABU=1               — disable visited_exprs tabu list
#
# Explicit CLI overrides vs. paper defaults:
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   --beam_width 40               (paper default: 20)
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_optionF_dedup_expr_rs
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

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
