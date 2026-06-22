#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v6_round2/memdiag_close
rm -rf $OUTDIR; mkdir -p $OUTDIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
WORKER=$BASE/scripts/eval/onestep_worker_v6.py
cat > $OUTDIR/memdiag.sub <<SUBEOF
universe = vanilla
executable = $PYTHON
arguments = -u $WORKER --topology $TOPOLOGY --integral='1,1,1,0,-2,1,1,1,0,0,0' --output $OUTDIR/result.pkl --model-checkpoint $MODEL --beam_width 40 --max_steps 1000 --prime 1009 --device cpu -v --no-paper-masters-only --checkpoint-path $OUTDIR/ckpt.pkl --checkpoint-interval 50 --checkpoint-time-seconds 300
environment = "PYTHONUNBUFFERED=1 SAILIR_MEM_BREAKDOWN=1 SAILIR_MEM_BREAKDOWN_EVERY=5"
output = $OUTDIR/memdiag.out
error  = $OUTDIR/memdiag.err
log    = $OUTDIR/memdiag.log
request_cpus = 1
request_memory = 30GB
request_disk = 40GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/memdiag.sub
