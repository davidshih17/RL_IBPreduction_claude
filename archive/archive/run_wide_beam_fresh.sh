#!/bin/bash
# Fresh attempt at the stuck (8,4) integral I[-1,2,1,0,1,2,1,1,-3,0,0]:
# start from SCRATCH (no resume — clear the 770-sub chain that the model
# can no longer fully see) with a WIDER beam (40 states instead of 20) and
# more parallel workers (16 CPUs). Uses the bitmask + cache A optimized
# code path that just shipped to source.
#
# Hypothesis: by step 770 the model only sees the last 50 subs and is
# wandering; from a fresh state with broader exploration (2x beam), it
# may find a weight-reducing path.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
SANDBOX=$BASE/results/wide_beam_fresh_v1
mkdir -p $SANDBOX/logs $SANDBOX/results

INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox

OUTPUT_PKL=$SANDBOX/results/result.pkl
CHECKPOINT=$SANDBOX/results/result.pkl.checkpoint

cat > $SANDBOX/run.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTPUT_PKL --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --n_workers 16 --checkpoint-path $CHECKPOINT --checkpoint-interval 50 --checkpoint-time-seconds 300
environment = "PYTHONUNBUFFERED=1"
output = $SANDBOX/logs/run.out
error = $SANDBOX/logs/run.err
log = $SANDBOX/logs/run.log
request_cpus = 16
request_memory = 32GB
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
echo "  grep 'Weight improved\|max_weight' $SANDBOX/logs/run.out | tail -5"
