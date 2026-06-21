#!/bin/bash
# v7 TOTAL-WEIGHT config on 4 probes: SUCCESS_TOTAL=1 (total-weight active-bucket
# success) + BEAM_TOTAL=1 (total-weight beam ranking + total-weight maxweight
# clip metric) + SAILIR_ACTION_SELECT=maxweight. Everything else v7 (all-(r,s)-
# tied target, (w1,w2) strip). beam 40, max-actions 900, beam-sort weight, 8/8.
# Compare step counts vs known v7 baselines: 74=81, 84=182, longrunner=968,
# memhog=720.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_BEAM_TOTAL=1 SAILIR_ACTION_SELECT=maxweight"

# name | integral | max_steps | mem(GB)
PROBES=(
  "74|0,1,1,1,1,1,1,1,-4,0,0|400|32"
  "84|-1,2,1,0,1,2,1,1,-3,0,0|600|32"
  "longrunner|1,1,1,0,0,1,3,1,-2,-1,0|2000|80"
  "memhog|1,1,1,0,-2,1,1,1,0,0,0|1500|64"
)
for row in "${PROBES[@]}"; do
  IFS='|' read -r name integ steps mem <<< "$row"
  D=$BASE/results/probe_${name}_v7_totalweight
  mkdir -p $D
  cat > $D/probe.sub <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/scripts/eval/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$integ' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 100 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps $steps --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 8
request_memory = ${mem}GB
request_disk = 80GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF
  condor_submit $D/probe.sub
done