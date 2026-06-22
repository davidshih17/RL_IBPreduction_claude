#!/bin/bash
# Re-run the long-runner async_4766 integral I[-1,-1,0,1,0,0,4,1,0,0,0] (weight
# (6,2)) with the FORK POOL OFF (--n-workers 1) but the model_fwd parallelization
# KEPT (--n-threads 8). Hypothesis: dramatically less memory (no 8-way COW spike
# -> ~main-process footprint, a few GB) for only a little more wall-time (the fork
# pool was measured worth ~14% on (7,4)).
#
# BASELINE (fork ON, async_4766 / cluster 1668448):
#   cgroup peak Memory = 22270 MB,  runtime 21583 s (~6h),  1772 steps, SUCCESS.
# Everything else EXACTLY matches the round3 / success-only probe recipe; the ONLY
# change is --n-workers 8 -> 1. Read the result from probe.log's terminated
# "Memory (MB) Usage" (cgroup max) and probe.out's t_total.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
INTEG='-1,-1,0,1,0,0,4,1,0,0,0'
D=$BASE/results/probe_nofork_4766
rm -rf $D; mkdir -p $D
# Identical env to the round3 recipe (NO SAILIR_KEEP_CKPT_EVERY — we only want
# the final memory+time, not per-step snapshots).
ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_ACTION_SELECT=maxweight"

cat > $D/probe.sub <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/scripts/eval/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$INTEG' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 100 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps 3000 --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads 8 --n-workers 1 --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 10
request_memory = 48GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $D/probe.sub
echo "fork-OFF probe submitted -> $D"
echo "  compare vs async_4766 (fork ON): 22270 MB / 21583 s / 1772 steps"
echo "  memory: grep 'Memory (MB)' $D/probe.log   (cgroup max at termination)"
echo "  time:   tail $D/probe.out                 (t_total + SUCCESS)"
