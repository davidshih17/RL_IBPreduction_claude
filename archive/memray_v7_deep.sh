#!/bin/bash
# memray attribution of the PACKED v7 memhog run, matching the memhog PROBE
# config (beam_search_v7.py + --tabu --iraws-keep-first 50 ...) so the
# attribution explains the ~7.5GB probe peak. From scratch (no resume) so
# resolved_subs is attributed to add_sub_to_resolved, not to unpickle/load.
# Compare report.txt against the v6 baseline:
#   results/pentagonbox_8_5_v6_round2/memray_deep/report.txt
#   (v6: compute_indirect_substituted_incremental = 17.76GB / 83.3%)
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/memray_v7_deep          # NEW dir (do not overwrite v6)
rm -rf $OUTDIR; mkdir -p $OUTDIR
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
INTEGRAL='1,1,1,0,-2,1,1,1,0,0,0'            # the memhog integral

cat > $OUTDIR/run_memray.sh <<WEOF
#!/bin/bash
set -e
cd $BASE/scripts/eval
$PY -m memray run --aggregate --trace-python-allocators --force \\
    -o $OUTDIR/capture.bin \\
    $SCRIPT --topology $TOPOLOGY --model $MODEL --integral='$INTEGRAL' \\
    --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 \\
    --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \\
    --max-steps 5000 --max-actions 900 --beam-sort weight \\
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \\
    --model-batch-chunk 8 > $OUTDIR/worker.log 2>&1
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
environment = "PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0"
request_cpus = 1
request_memory = 44GB
request_disk = 60GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
SUBEOF
condor_submit $OUTDIR/memray.sub
