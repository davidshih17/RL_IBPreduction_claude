#!/bin/bash
# Experiment: does a different ACTION-CAP strategy let the stuck (7,4) seed
# I[-4,2,1,0,1,1,1,1,0,0,0] escape the (7,4) shell and produce below-start
# passengers (which maxweight never did)? Submit two probes identical to the
# round3 recipe EXCEPT SAILIR_ACTION_SELECT: 'sumweight' and 'shortest'.
# Versioned beam snapshots every 10 steps (SAILIR_KEEP_CKPT_EVERY=10) so
# compare_totalweight.py can classify each step's non-masters ABOVE/BELOW start.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
INTEG='-4,2,1,0,1,1,1,1,0,0,0'
ENVBASE="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_KEEP_CKPT_EVERY=10"

# rows: "label|SAILIR_ACTION_SELECT|max_actions"
PROBES=(
  "sumweight|sumweight|900"
  "shortest|shortest|900"
  "maxweight_ac2000|maxweight|2000"   # maxweight but with a wider action window
)
for row in "${PROBES[@]}"; do
  IFS='|' read -r LABEL STRAT MAXACT <<< "$row"
  D=$BASE/results/probe_stuck_74_${LABEL}
  rm -rf $D; mkdir -p $D
  ENV="$ENVBASE SAILIR_ACTION_SELECT=$STRAT"
  cat > $D/probe.sub <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/scripts/eval/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$INTEG' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 50 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 600 --max-actions $MAXACT --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 10
request_memory = 24GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF
  condor_submit $D/probe.sub
  echo "submitted $LABEL (ACTION_SELECT=$STRAT max-actions=$MAXACT) -> $D"
done
echo ""
echo "watch:   results/probe_stuck_74_{sumweight,shortest,maxweight_ac2000}/probe.out"
echo "snaps:   results/probe_stuck_74_*/ckpt.pkl.keep_step000NN"
echo "analyze: python scripts/eval/compare_totalweight.py <dir>"
