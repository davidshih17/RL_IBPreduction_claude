#!/bin/bash
# Stop the running (8,5) orchestrator + its workers cleanly, then restart
# with --resume to reload all 74k worker pickles as cache, plus enable the
# new --worker-dedup-beam-by-content flag so future worker submissions use
# the verified-correct resolved_subs-based dedup.
#
# Run this on the login node where the orchestrator started.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v3

echo "=== Step 1: condor_rm only the production (8,5) worker jobs ==="
# Filter: any job whose Args contain "pentagonbox_8_5_v3/work" is a production
# worker submitted by the orchestrator. Leaves our experimental jobs alone.
PROD_CIDS=$(condor_q dshih -af ClusterId Args 2>&1 \
    | grep "pentagonbox_8_5_v3/work" \
    | awk '{print $1}' \
    | sort -un)
N_PROD=$(echo "$PROD_CIDS" | wc -w)
echo "  Found $N_PROD production worker clusters to remove"
if [[ -n "$PROD_CIDS" ]]; then
    # condor_rm in chunks (10000-arg limit safety)
    echo "$PROD_CIDS" | xargs -n 1000 condor_rm 2>&1 | tail -3
fi
echo "  (wait briefly for queue to drain)"
sleep 10
echo "  Remaining dshih jobs:"
condor_q dshih -totals 2>&1 | tail -2

echo
echo "=== Step 2: kill the orchestrator process ==="
ORCH_PID=$(ps -u dshih -o pid,cmd 2>&1 | grep "hierarchical_reduction.py" | grep -v grep | awk '{print $1}')
echo "  orchestrator PID: $ORCH_PID"
if [[ -n "$ORCH_PID" ]]; then
    kill $ORCH_PID
    sleep 3
    if ps -p $ORCH_PID > /dev/null; then
        echo "  process still alive, sending SIGKILL"
        kill -9 $ORCH_PID
    fi
    echo "  killed."
fi

echo
echo "=== Step 3: relaunch with --resume + --worker-dedup-beam-by-content ==="
# Save the OLD log so we don't overwrite it.
mv $OUTDIR/logs/hierarchical.log $OUTDIR/logs/hierarchical.log.pre_resume_$(date +%s)

cd $BASE
export PYTHONUNBUFFERED=1
nohup /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
    $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $BASE/topology_input/pentagonbox \
    --integral 1,1,1,1,1,1,1,1,-5,0,0 \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
    --beam_width 20 --max_steps 1000000 --prime 1009 \
    --straggler-timeout 3600 \
    --straggler-cpus 8 \
    --checkpoint-interval 50 \
    --checkpoint-time-seconds 300 \
    --resume \
    > $OUTDIR/logs/hierarchical.log 2>&1 &
    # NOTE: worker dedup is now ON by default; --no-worker-dedup-beam-by-content disables.

echo "  orchestrator started, PID $!"
echo "  watch: tail -f $OUTDIR/logs/hierarchical.log"
