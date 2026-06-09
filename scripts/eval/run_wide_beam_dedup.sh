#!/bin/bash
# Fresh attempt at the stuck (8,4) integral with WIDE beam (40)
# + content-based deduplication. The dedup keeps only the first occurrence
# of each unique target-sector expr content in the beam, so the 40 slots
# carry 40 GENUINELY distinct exprs instead of ~7 unique + 33 duplicates.
# Hypothesis: the model's "wandering" got amplified because the beam was
# stuffed with near-duplicates that all explored the same wrong direction.
# Real diversity might find the way out.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
SANDBOX=$BASE/results/wide_beam_dedup_v1
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
