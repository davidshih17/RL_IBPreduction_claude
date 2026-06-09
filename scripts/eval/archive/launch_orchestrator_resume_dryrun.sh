#!/bin/bash
# Dry-run of orchestrator with new checkpoint+resume.
# Target: pentagonbox_first (4,2). Forces straggler resubmit early so we
# verify the 8-CPU resubmit actually resumes from the killed worker's
# checkpoint.
#
#   straggler-timeout = 120 s  (2 min, so 1-CPU workers get killed quickly)
#   checkpoint-time-seconds = 30 s (so at least one checkpoint is saved
#                                   before the 2-min kill)
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/orchestrator_dryrun_resume
mkdir -p $OUTDIR/logs $OUTDIR/work

cd $BASE

export PYTHONUNBUFFERED=1
/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 1,0,1,0,1,1,-1,0,0,0,-1 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
    --beam_width 20 --max_steps 1000000 --prime 1009 \
    --straggler-timeout 120 \
    --straggler-cpus 8 \
    --checkpoint-interval 50 \
    --checkpoint-time-seconds 30 \
    > $OUTDIR/logs/hierarchical.log 2>&1
