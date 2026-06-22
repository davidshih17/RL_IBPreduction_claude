#!/bin/bash
# v4: bounded-anchor delta with useful_K ONLY (drop expr_nm anchors).
# Phase 1a handles direct actions per target; Phase 1b iraws only needs
# past-sub_int K's where sol_K ∩ expr_nm ≠ ∅. Matches baseline iraws
# structure exactly, just pruned to useful entries.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_exprkeyed_v4
mkdir -p $OUTDIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/delta_onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 1
environment = "PYTHONUNBUFFERED=1 DELTA_IRAWS_EXPRKEYED=1 DELTA_REBUILD_INTERVAL=0 DELTA_SECTOR_PROJECT_AUX=0"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 32GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
