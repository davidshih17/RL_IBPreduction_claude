#!/bin/bash
# Full (8,5) hierarchical reduction with v6 workers.
# Mirrors the failed delta-worker sweep at pentagonbox_8_5_delta but
# dispatches onestep_worker_v6.py underneath. v6 was 9.7x faster and
# 10.9x lower peak memory than delta on the canonical long-runner; the
# (8,5) sweep is the real-world test.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v6
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"

# Orchestrator runs IN PROCESS (it's just a poller — submits + collects
# Condor jobs). Send it to the background and stream stdout/stderr to
# the master log so we can monitor without blocking the shell.
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v6-worker \
    --worker-memory-gb 16 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 10000 \
    --resume \
  > $OUTDIR/logs/hierarchical.log 2>&1 &

ORCH_PID=$!
echo "Orchestrator launched PID=$ORCH_PID"
echo "  log:    $OUTDIR/logs/hierarchical.log"
echo "  workdir: $OUTDIR/work"
echo "  result: $OUTDIR/reduction.pkl"
