#!/bin/bash
# (8,4) probe: DELTA_P4_WORKERS=16 + DELTA_P4_INSTR=1, NEW concurrent pipe drain.
# Apples-to-apples vs probe_84_delta_p4instr (sequential drain). Should show
# pipe_read dropping from ~2.1s to under 0.3s; total P4 from ~5s to ~2s.
# All other parallelization is off so the P4 effect is isolated.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_p4drain
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/delta_onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 16
environment = "PYTHONUNBUFFERED=1 DELTA_P4_WORKERS=16 DELTA_P4_INSTR=1"
output = $OUTDIR/probe.out
error  = $OUTDIR/probe.err
log    = $OUTDIR/probe.log
request_cpus = 16
request_memory = 32GB
request_disk = 1GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit $OUTDIR/probe.sub
