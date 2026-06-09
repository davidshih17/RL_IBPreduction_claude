#!/bin/bash
# Run the bitmask-optimized reduction on the slowest production integral
# (straggler_19642 = I[-1,2,1,0,1,2,1,1,-3,0,0]), resumed from the latest
# production checkpoint. Goal: see whether the integral EVER finds a weight
# improvement, given the ~5x P1 speedup from the bitmask optimization.
#
# No BEAM_PROFILE_CSV (we want full speed). Output to its own sandbox so it
# doesn't disturb production.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PROD=$BASE/results/pentagonbox_8_5_v3/work/results
SANDBOX=$BASE/results/optimized_run_v1
mkdir -p $SANDBOX/logs $SANDBOX/results

INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"
PROD_CHECKPOINT=$PROD/straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint
SANDBOX_CHECKPOINT=$SANDBOX/resumed.checkpoint
cp -f $PROD_CHECKPOINT $SANDBOX_CHECKPOINT
echo "Copied $(ls -lh $SANDBOX_CHECKPOINT | awk '{print $5}') checkpoint to sandbox"

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox

# Read the step from the copied snapshot so max_steps is meaningful.
RESUMED_STEP=$($PYTHON -c "
import pickle, sys
sys.path.insert(0, '$BASE/scripts/eval')
with open('$SANDBOX_CHECKPOINT','rb') as f: ck = pickle.load(f)
print(int(ck['step']))
")
# Allow up to +2000 steps from resume so the worker has a real shot at
# finding weight improvement. The internal stop_on_weight_improvement
# logic returns as soon as a weight reduction is found, so this is just an
# upper bound — if it never improves we'll see how far it gets.
MAX_STEPS=$((RESUMED_STEP + 2000))
echo "Resumed step=$RESUMED_STEP; will run to step $MAX_STEPS (+2000 steps max)"

OUTPUT_PKL=$SANDBOX/results/profile.pkl
NEW_CHECKPOINT=$SANDBOX/results/profile.pkl.checkpoint

# 4 CPUs matches the slot pool we know we can get a slot in. (Production
# uses 8, but 4 + bitmask is still faster than 8 + old code by ~2x.) 16 GB
# memory is sized to comfortably hold the indirect_cache growth across
# many steps.
cat > $SANDBOX/run.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTPUT_PKL --model-checkpoint $MODEL --beam_width 20 --max_steps $MAX_STEPS --prime 1009 --device cpu -v --n_workers 4 --checkpoint-path $NEW_CHECKPOINT --checkpoint-interval 50 --checkpoint-time-seconds 300 --resume-from $SANDBOX_CHECKPOINT
environment = "PYTHONUNBUFFERED=1"
output = $SANDBOX/logs/run.out
error = $SANDBOX/logs/run.err
log = $SANDBOX/logs/run.log
request_cpus = 4
request_memory = 12GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $SANDBOX/run.sub
echo
echo "Run submitted. Watch progress with:"
echo "  tail -f $SANDBOX/logs/run.out"
echo "  grep 'Weight improved' $SANDBOX/logs/run.out"
echo "  grep '^Step ' $SANDBOX/logs/run.out | tail -5"
