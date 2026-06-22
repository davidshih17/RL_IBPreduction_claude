#!/bin/bash
# v7 with ONLY the total-weight SUCCESS criterion reverted-in; beam ranking and
# maxweight action-ordering BACK to (r,s) (i.e. SAILIR_BEAM_TOTAL is NOT set).
# Config = old 74_maxweight + total-weight success. 5 probes, beam-sort weight,
# beam 40, max-actions 900, maxweight clip, 8 threads/8 workers (affinity pin
# keeps CPU <= 8). Baselines: 74=81, 84=182, longrunner=968, memhog=720,
# monster=never(23h/1596). Total-weight runs (with BEAM_TOTAL) were 84=8/lr=1/
# memhog=67/monster=107 -- this tests whether SUCCESS_TOTAL ALONE keeps the win.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
# NOTE: SAILIR_BEAM_TOTAL deliberately ABSENT -> (r,s) beam + (r,s) maxweight.
ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_ACTION_SELECT=maxweight"

PROBES=(
  "74|0,1,1,1,1,1,1,1,-4,0,0|400|32"
  "84|-1,2,1,0,1,2,1,1,-3,0,0|600|32"
  "longrunner|1,1,1,0,0,1,3,1,-2,-1,0|2000|80"
  "memhog|1,1,1,0,-2,1,1,1,0,0,0|1500|64"
  "monster|-1,1,1,0,1,1,0,1,-3,0,0|3000|96"
)
for row in "${PROBES[@]}"; do
  IFS='|' read -r name integ steps mem <<< "$row"
  D=$BASE/results/probe_${name}_v7_successonly
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