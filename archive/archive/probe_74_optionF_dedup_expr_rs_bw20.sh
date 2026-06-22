#!/bin/bash
# (7,4) probe — Option-F + DEDUP_VARIANT=expr_rs + NO_TABU=1, beam_width=20.
#
# Sibling of probe_74_optionF_dedup_expr_rs.sh, ONLY change: bw 40 -> 20.
# Tests whether the paper-default beam width is enough to crack (7,4)
# under the expr_rs dedup that bit-identically cracked (8,4) at bw=40
# (cluster 1478843 cracked (7,4) at step 342 in 2533s with bw=40).
#
# Explicit CLI overrides vs. paper defaults:
#   --beam-sort weight            (paper default: mixed)
#   --dedup-beam-by-content       (paper default: off)
#   --no-paper-masters-only       (paper default: on)
#   (beam_width left at paper default: 20)
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_optionF_dedup_expr_rs_bw20
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 20 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-interval 50 --checkpoint-time-seconds 300 --beam-sort weight --dedup-beam-by-content --no-paper-masters-only
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
