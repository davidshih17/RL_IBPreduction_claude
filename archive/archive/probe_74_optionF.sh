#!/bin/bash
# Probe (7,4) with Option F (target-only substitution + path-replay reconstruction).
#
# Identical configuration to probe_74_no_dedup_with_ckpt.sh — same model, same
# topology, same integral, same beam_width, same dedup-off, same checkpoint
# settings. The only difference is the underlying code path:
#   - Baseline (probe_74_no_dedup_with_ckpt, cluster 1468428): split-expr with
#     sub_accum tracked through every step.
#   - This run (Option F): sub-sector terms DISCARDED during beam search and
#     reconstructed at worker end by replaying path through subs against the
#     original start_expr.
#
# Expected: bit-identical full final_expr (target-sector AND sub-sector content).
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_optionF
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 20 --max_steps 1000000 --prime 1009 --device cpu -v --n_workers 8 --no-dedup-beam-by-content --checkpoint-interval 25 --checkpoint-time-seconds 300
environment = "PYTHONUNBUFFERED=1"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 8
request_memory = 16GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
