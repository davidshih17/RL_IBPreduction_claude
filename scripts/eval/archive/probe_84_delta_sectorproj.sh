#!/bin/bash
# (8,4) probe: DELTA_SECTOR_PROJECT_AUX=1 — drop outside-target-sector keys from
# every cached dict in cu (both Phase A and Phase B of incremental aux, plus
# R1 full rebuild).
# Goal: enforce the "work modulo subsectors EVERYWHERE in the worker" principle
#       and measure memory reduction. Also verify the reduction reaches masters.
#
# DELTA_REBUILD_INTERVAL=50 (R1) still ON.
# Compare to:
#   - rebuild50-only (no sector projection): 5058s, 12.4 GB peak, 261 steps
#   - baseline p4drain:                       5491s, ~16 GB peak,  261 steps
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_sectorproj
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/delta_onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 1
environment = "PYTHONUNBUFFERED=1 DELTA_REBUILD_INTERVAL=50 DELTA_SECTOR_PROJECT_AUX=1"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 1
request_memory = 32GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
