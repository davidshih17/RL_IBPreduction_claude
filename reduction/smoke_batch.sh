#!/bin/bash
# Smoke-test the BATCHED submit/cancel orchestrator on a small fresh reduction.
# Small (4,2) sub-sector target -> few iterations, light schedd load. Verifies:
#   - create_batch_submit + submit_condor_batch (one condor_submit -> one cluster
#     with procs 0..N-1), batched obsolete-cancel, cluster.proc tracking
#   - the reduction still reaches SUCCESS (correctness unaffected by batching)
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/_batch_smoke
rm -rf $OUTDIR
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,0,1,1,1,0,0,1,-2,0,0"     # weight (4,2)

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL_STR" \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --check-interval 5 --max-concurrent 1000 --resume \
  > $OUTDIR/logs/hierarchical.log 2>&1 &
echo "smoke PID=$!  log: $OUTDIR/logs/hierarchical.log"
