#!/bin/bash
# Probe the stuck (7,4) round3 target I[-4,2,1,0,1,1,1,1,0,0,0] (worker 1663961,
# best mw=(7,4) nm=1 churning for 1800+ steps). Runs beam_search_v7 with the
# EXACT round3/probe settings (so the trajectory is bit-identical to the stuck
# worker), but capped at 200 steps and with SAILIR_KEEP_CKPT_EVERY=5 so it writes
# a VERSIONED beam snapshot (<ckpt>.keep_step00005, .keep_step00010, ...) every 5
# steps and KEEPS them all. inspect_keep_ckpts.py then loads each snapshot's
# best-state expr to see whether the (7,4) nm=1 integral is the SAME one recurring
# or different ones cycling.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
INTEG='-4,2,1,0,1,1,1,1,0,0,0'
D=$BASE/results/probe_stuck_74_ckpt
rm -rf $D; mkdir -p $D
# EXACT round3 env (submit_v7_successonly_probes.sh) + keep-every-5 versioned ckpts.
ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_ACTION_SELECT=maxweight SAILIR_KEEP_CKPT_EVERY=5"

cat > $D/probe.sub <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/scripts/eval/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$INTEG' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 200 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 10
request_memory = 16GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $D/probe.sub
echo "probe submitted. dir=$D"
echo "  keep-step snapshots will appear as: $D/ckpt.pkl.keep_step000NN"
echo "  watch: $D/probe.out  ;  inspect with inspect_keep_ckpts.py after a few min"
