#!/bin/bash
# FRESH full (8,5) hierarchical reduction with the v7 worker at 1 CPU.
#
# This is a from-scratch reduction of the (8,5) pentagonbox top integral
# I[1,1,1,1,1,1,1,1,-5,0,0] -- NO --resume-from, NO --reduce-only. The
# orchestrator enumerates the start, dispatches one Condor worker per
# non-master, and chains weight levels down to the masters, exactly like
# round1 (launch_hier_85_v6.sh).
#
# The ONLY change vs the successful rounds 3+4 (which used --v7-cpus 8,
# request_cpus=10, 8/8 fork pool + torch threads, ~22 GB peak per worker):
#   --v7-cpus 1  ->  serial enumerate (NO os.fork, NO fork pool), torch
#                    single-threaded, affinity pinned to 1 core,
#                    request_cpus=1, flat 4 GB request.
# All v7 ALGORITHM settings are identical to rounds 3+4 (SUCCESS_TOTAL
# total-weight single-step success, (r,s) maxweight action-cap,
# beam_width 40, prime 1009, --no-paper-masters-only).
#
# Output goes to a FRESH dir so the previous successful reduction
# (results/pentagonbox_8_5_v6_round{1..4}) is NOT touched.
#
# Goal: run overnight with up to 1000 cheap 1-CPU/4 GB workers and see how
# far the reduction gets without the fork-pool memory blow-up.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v7_fresh
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"

# Orchestrator runs IN PROCESS (it's just a poller -- submits + collects
# Condor jobs). Send it to the background and stream stdout+stderr into a
# single master log (unbuffered) so we can monitor without blocking.
set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $OUTDIR/logs/hierarchical_fresh.log 2>&1 &
ORCH_PID=$!
set +x
echo "FRESH v7 1-cpu orchestrator launched PID=$ORCH_PID"
echo "  log:     $OUTDIR/logs/hierarchical_fresh.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
