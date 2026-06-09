#!/bin/bash
# Probe (7,4) with Option F + resolved_subs storage stripping + IBP-template
# replay at worker end.
#
# Same configuration as probe_74_no_dedup_with_ckpt.sh and probe_74_optionF.sh
# (no dedup, same beam_width, same model). The only difference vs the previous
# Option F run is the new algorithmic refinement:
#   - subs[target] and resolved_subs[target] values stripped to target-sector
#     at storage time (cheap dedup hashing, smaller apply_resolved_subs work)
#   - full final_expr reconstructed at worker end by replaying path through
#     raw IBP templates (env.replay_path_to_full_expr)
#
# Expected: bit-identical full final_expr vs cluster 1468428 baseline.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_optionF_stripped
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
