#!/bin/bash
# 4 probes x 5 action-select strategies at the optimal 8/8 + kernel config.
# Each job runs to SUCCESS (or max-steps), recording steps-to-success.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
ROOT=$BASE/results/actionselect_exp
mkdir -p $ROOT

# probe name : integral : max_steps : request_memory
declare -A INT=( [74]="0,1,1,1,1,1,1,1,-4,0,0" [84]="-1,2,1,0,1,2,1,1,-3,0,0" \
                 [long]="1,1,1,0,0,1,3,1,-2,-1,0" [memhog]="1,1,1,0,-2,1,1,1,0,0,0" )
declare -A MAXS=( [74]=600 [84]=900 [long]=2500 [memhog]=2000 )
declare -A MEM=(  [74]="8GB" [84]="8GB" [long]="48GB" [memhog]="48GB" )

for probe in 74 84 long memhog; do
  for strat in first900 last900 maxweight shortest sumweight; do
    O=$ROOT/${probe}_${strat}; mkdir -p $O
    cat > $O/probe.sub <<SUBEOF
universe = vanilla
executable = $PYTHON
arguments = -u $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='${INT[$probe]}' --output $O/result.pkl --ckpt $O/ckpt.pkl --ckpt-every 100 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps ${MAXS[$probe]} --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8
environment = "PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_ACTION_SELECT=${strat}"
output = $O/probe.out
error  = $O/probe.err
log    = $O/probe.log
request_cpus = 8
request_memory = ${MEM[$probe]}
request_disk = 50GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
    condor_submit $O/probe.sub > /dev/null 2>&1 && echo "submitted ${probe}_${strat}"
  done
done
echo "=== all 20 submitted ==="
