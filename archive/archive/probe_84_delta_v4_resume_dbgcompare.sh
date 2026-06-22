#!/bin/bash
# Resume from v4's step-214 thin checkpoint with DELTA_DEBUG_COMPARE=1. The
# resume replays each of 40 beam survivors' 214-step paths in serial mode
# (no model inference, no beam selection), triggering _materialize_aux for
# every replayed step. My compare diffs v4-incremental iraws against a
# from-scratch baseline rebuild and prints the first iraws divergence.
#
# Expected: catches the iraws-level root cause that led to the survivor-
# level divergence observed at step 153 in diff_thin_checkpoints.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
SRC_CKPT=$BASE/results/probe_84_delta_v4_ckpt/result.pkl.ckpt.r1.step214
OUTDIR=$BASE/results/probe_84_delta_v4_resume_dbgcompare
mkdir -p $OUTDIR
cp $SRC_CKPT $OUTDIR/result.pkl.ckpt.r1
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/delta_onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 1
environment = "PYTHONUNBUFFERED=1 DELTA_IRAWS_EXPRKEYED=1 DELTA_DEBUG_COMPARE=1 DELTA_REBUILD_INTERVAL=0 DELTA_SECTOR_PROJECT_AUX=0"
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
