#!/bin/bash
# A/B on Condor: v7 baseline ((w1,w2) active-bucket success) vs v7 with
# SAILIR_SUCCESS_TOTAL=1 (total-weight active-bucket success). ONLY difference is
# the success criterion -> expect the total run to finish in <= the baseline's
# steps (strictly weaker). probe 74, beam 40, max-actions 900, 8/8.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
ARGS="--topology $TOPO --model $MODEL --integral='0,1,1,1,1,1,1,1,-4,0,0' --ckpt-every 100 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 200 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8"
BASEENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1"

for tag in base total; do
  D=$BASE/results/probe_74_v7_${tag}
  mkdir -p $D
  ENV="$BASEENV"
  [ "$tag" = "total" ] && ENV="$BASEENV SAILIR_SUCCESS_TOTAL=1"
  cat > $D/probe.sub <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/scripts/eval/beam_search_v7.py $ARGS --output $D/result.pkl --ckpt $D/ckpt.pkl
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 8
request_memory = 32GB
request_disk = 50GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF
  condor_submit $D/probe.sub
done