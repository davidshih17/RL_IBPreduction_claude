#!/bin/bash
# Profile beam_search internals on a slow-stage checkpoint copied from the
# live (8,5) production run. The instrumentation in beam_search.py writes a
# per-state, per-step CSV when BEAM_PROFILE_CSV is set in the worker's env.
#
# We copy the production checkpoint (don't touch the original) and submit a
# single 8-CPU Condor job with priority 1e9 so it jumps the 10k-job queue.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PROD=$BASE/results/pentagonbox_8_5_v3/work/results
SANDBOX=$BASE/results/profile_sandbox_v1
mkdir -p $SANDBOX/logs $SANDBOX/results

# Slowest-stepping 8-CPU straggler from the latest survey
# (cluster 1399241, step 612, P1=285-385s, integral -1,2,1,0,1,2,1,1,-3,0,0).
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"
PROD_CHECKPOINT=$PROD/straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint
if [[ ! -f $PROD_CHECKPOINT ]]; then
    echo "ERROR: production checkpoint not found at $PROD_CHECKPOINT" >&2
    exit 1
fi

# Atomic copy — beam_search writes via .tmp + rename, so a torn read is
# impossible; we either get the prior snapshot or the current one.
SANDBOX_CHECKPOINT=$SANDBOX/resumed.checkpoint
cp -f $PROD_CHECKPOINT $SANDBOX_CHECKPOINT
echo "Copied $(ls -lh $SANDBOX_CHECKPOINT | awk '{print $5}') checkpoint to sandbox"

PROFILE_CSV=$SANDBOX/profile.csv
OUTPUT_PKL=$SANDBOX/results/profile.pkl
NEW_CHECKPOINT=$SANDBOX/results/profile.pkl.checkpoint

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox

# Peek at the checkpoint's saved step so we can set max_steps = step+30.
# beam_search uses absolute step counts, so --max_steps must exceed the
# resumed step or the loop exits immediately. The checkpoint references the
# beam_search.State class, so we must add scripts/eval to sys.path before
# unpickling.
RESUMED_STEP=$($PYTHON -c "
import pickle, sys
sys.path.insert(0, '$BASE/scripts/eval')
with open('$SANDBOX_CHECKPOINT','rb') as f: ck = pickle.load(f)
print(int(ck['step']))
")
MAX_STEPS=$((RESUMED_STEP + 2))
echo "Checkpoint saved at step $RESUMED_STEP; will run to step $MAX_STEPS (+2 instrumented steps, avoid OOM)"

# Submit file. priority=1_000_000_000 (1B) jumps ahead of the 10k queued
# production jobs whose priorities are in the 500-8M range.
cat > $SANDBOX/profile.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTPUT_PKL --model-checkpoint $MODEL --beam_width 20 --max_steps $MAX_STEPS --prime 1009 --device cpu -v --n_workers 4 --checkpoint-path $NEW_CHECKPOINT --checkpoint-interval 50 --checkpoint-time-seconds 300 --resume-from $SANDBOX_CHECKPOINT
environment = "BEAM_PROFILE_CSV=$PROFILE_CSV PYTHONUNBUFFERED=1"
output = $SANDBOX/logs/profile.out
error = $SANDBOX/logs/profile.err
log = $SANDBOX/logs/profile.log
request_cpus = 4
request_memory = 12GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $SANDBOX/profile.sub

echo
echo "Profile job submitted. Monitor with:"
echo "  condor_q dshih"
echo "  tail -f $SANDBOX/logs/profile.out"
echo "  watch -n 5 'wc -l $PROFILE_CSV'"
echo
echo "CSV will appear at: $PROFILE_CSV"
