#!/bin/bash
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/pentagonbox_8_5_v6_round2/memray_deep
rm -rf $OUTDIR; mkdir -p $OUTDIR
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
WORKER=$BASE/scripts/eval/onestep_worker_v6.py
cat > $OUTDIR/run_memray.sh <<WEOF
#!/bin/bash
set -e
cd $BASE/scripts/eval
$PY -m memray run --aggregate --trace-python-allocators --force \
    -o $OUTDIR/capture.bin \
    $WORKER --topology $TOPOLOGY --integral='1,1,1,0,-2,1,1,1,0,0,0' \
    --output $OUTDIR/result.pkl --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000 --prime 1009 --device cpu -v \
    --no-paper-masters-only > $OUTDIR/worker.log 2>&1
$PY -m memray summary $OUTDIR/capture.bin > $OUTDIR/report.txt 2>&1 || true
echo DONE >> $OUTDIR/report.txt
WEOF
chmod +x $OUTDIR/run_memray.sh
cat > $OUTDIR/memray.sub <<SUBEOF
universe = vanilla
executable = $OUTDIR/run_memray.sh
output = $OUTDIR/memray.out
error  = $OUTDIR/memray.err
log    = $OUTDIR/memray.log
environment = "PYTHONUNBUFFERED=1"
request_cpus = 1
request_memory = 44GB
request_disk = 60GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/memray.sub
