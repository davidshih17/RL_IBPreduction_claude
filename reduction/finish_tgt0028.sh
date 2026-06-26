#!/bin/bash
# tgt0028's surviving workers (cluster 1830090) finished 11/12 non-masters before
# the crash-resume mixup; 1 is still running. Wait for it to drain (so we don't
# re-duplicate it), THEN re-resume the orchestrator ONCE -- by then all 12 results
# are on disk, so it just collects them and writes reduction.pkl (no new submits).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
CACHE=$BASE/results/meta_reduce/_resume_cache
OUTDIR=$BASE/results/meta_reduce/tgt0028_w7_4

echo "waiting for tgt0028's last survivor worker to drain..."
while [ "$(condor_q -nobatch 2>/dev/null | grep -c 'tgt0028')" -gt 0 ]; do sleep 60; done
echo "drained at $(date '+%H:%M:%S'); re-resuming tgt0028 to collect results"

PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY --integral="1,1,1,1,1,1,-1,1,-2,0,-1" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --resume-from $CACHE --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --check-interval 5 --max-concurrent 1000 --resume \
    > $OUTDIR/logs/resume2.log 2>&1
echo "tgt0028 re-resume finished: $(grep -E 'SUCCESS|All integrals reduced' $OUTDIR/logs/resume2.log | tail -1)"
