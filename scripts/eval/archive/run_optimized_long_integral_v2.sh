#!/bin/bash
# v2: resume from optimized_run_v1's checkpoint (step ~768, weight (8,4)) using
# the bitmask filter + cache A code path. Uses 8 CPUs for apples-to-apples
# comparison against the production baseline.
#
# Goal: see how much faster the (8,4) integral chews through steps with the
# combined optimizations vs the (bitmask-only) v1 run.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PREV=$BASE/results/optimized_run_v1
SANDBOX=$BASE/results/optimized_run_v2
mkdir -p $SANDBOX/logs $SANDBOX/results

INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"
PREV_CHECKPOINT=$PREV/results/profile.pkl.checkpoint
SANDBOX_CHECKPOINT=$SANDBOX/resumed.checkpoint

if [[ ! -f $PREV_CHECKPOINT ]]; then
    echo "ERROR: previous checkpoint not found at $PREV_CHECKPOINT" >&2
    exit 1
fi
cp -f $PREV_CHECKPOINT $SANDBOX_CHECKPOINT
echo "Copied $(ls -lh $SANDBOX_CHECKPOINT | awk '{print $5}') from optimized_run_v1"

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox

RESUMED_STEP=$($PYTHON -c "
import pickle, sys
sys.path.insert(0, '$BASE/scripts/eval')
with open('$SANDBOX_CHECKPOINT','rb') as f: ck = pickle.load(f)
print(int(ck['step']))
")
MAX_STEPS=$((RESUMED_STEP + 2000))
echo "Resuming at step $RESUMED_STEP; will run to step $MAX_STEPS"

OUTPUT_PKL=$SANDBOX/results/profile.pkl
NEW_CHECKPOINT=$SANDBOX/results/profile.pkl.checkpoint

cat > $SANDBOX/run.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTPUT_PKL --model-checkpoint $MODEL --beam_width 20 --max_steps $MAX_STEPS --prime 1009 --device cpu -v --n_workers 8 --checkpoint-path $NEW_CHECKPOINT --checkpoint-interval 50 --checkpoint-time-seconds 300 --resume-from $SANDBOX_CHECKPOINT
environment = "PYTHONUNBUFFERED=1"
output = $SANDBOX/logs/run.out
error = $SANDBOX/logs/run.err
log = $SANDBOX/logs/run.log
request_cpus = 8
request_memory = 16GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $SANDBOX/run.sub
echo
echo "Watch with:"
echo "  tail -f $SANDBOX/logs/run.out"
echo "  grep '^Step ' $SANDBOX/logs/run.out | tail -5"
