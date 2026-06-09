#!/bin/bash
# (8,4) probe — Option-F + expr_rs dedup + NO_TABU + beam_width 20 +
# parent-quota M=2 (no single parent in the previous beam contributes
# more than 2 of the 20 next-beam slots → at least 10 distinct parents).
#
# Companion to probe_74_optionF_dedup_expr_rs_bw20_pq2.sh.
# bw=40 expr_rs cracks (8,4) bit-identically to wide_beam_dedup_v1
# (cluster 1478829, 261 steps, 2506s). bw=20 expr_rs alone was running
# 332+ steps without success (cluster 1479039). This adds parent quota
# to test whether the bw=40 success was diversity-driven.
#
# Selectors:
#   DEDUP_VARIANT=expr_rs
#   NO_TABU=1
#   MAX_CHILDREN_PER_PARENT=2
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_optionF_dedup_expr_rs_bw20_pq2
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 20 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
environment = "PYTHONUNBUFFERED=1 DEDUP_VARIANT=expr_rs NO_TABU=1 MAX_CHILDREN_PER_PARENT=2"
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
