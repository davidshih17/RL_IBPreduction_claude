#!/bin/bash
# Probe (7,4) with Option F + storage stripping + dedup-by-expr + TABU
# list of visited exprs. Tabu: a candidate whose expr was in any prior
# step's beam is pushed to leftovers (and only used to pad the beam if
# nothing else is available). This should break the expr cycles we
# observed under plain dedup-by-expr.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_74_dedup_tabu
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="0,1,1,1,1,1,1,1,-4,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = $PYTHON
arguments = -u $BASE/scripts/eval/onestep_worker.py --topology $TOPOLOGY --integral='$INTEGRAL_STR' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 20 --max_steps 1000000 --prime 1009 --device cpu -v --n_workers 8 --checkpoint-interval 25 --checkpoint-time-seconds 300
environment = "PYTHONUNBUFFERED=1 BEAM_DUMP_FULL=$OUTDIR/beam.log"
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
