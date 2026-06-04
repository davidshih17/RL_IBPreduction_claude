#!/bin/bash
# (8,5) pentagon-box reduction via the delta-tracking serial worker.
#
# Settings per user request (2026-05-31):
#   --use-delta-worker       : dispatch to delta_onestep_worker.py
#                              (serial 1-cpu + Cython phase1b/Phase A/build_result)
#   --straggler-timeout / --straggler2-timeout = 10^9 sec
#                              : effectively disables the straggler escalation
#                                workflow. Every worker runs at 1 CPU
#                                indefinitely.
#   --worker-memory-gb 16    : probe_84_delta peaked at 10.6 GB; 16 leaves
#                              headroom and matches far more slots than 32
#                              (the cluster has many 8-16 GB slots).
#   Priority order is already (level, r, s) lex via job_priority formula
#   in hierarchical_reduction.py:147.
#   --paper-masters-only / --no-paper-masters-only: NOT passed → orchestrator
#                              default ON. Matches probe_84_p3instr.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_delta
mkdir -p $OUTDIR/logs $OUTDIR/work

cd $BASE

export PYTHONUNBUFFERED=1
nohup /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 1,1,1,1,1,1,1,1,-5,0,0 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-delta-worker \
    --worker-memory-gb 16 \
    --resume \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    > $OUTDIR/logs/hierarchical.log 2>&1 &
echo "Launched as pid $!; log: $OUTDIR/logs/hierarchical.log"
