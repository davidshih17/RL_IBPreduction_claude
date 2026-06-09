#!/bin/bash
# (8,4) probe: TEST THIN checkpoint (path-only) save + replay-resume.
# Stage 1: run --max_steps 100 with --checkpoint --checkpoint-mode thin
#          -> expect KB-scale ckpt files, not yet at masters
# Stage 2: run --resume + --max_steps 5000
#          -> should replay paths to rebuild state, then continue to completion.
# Verify: size of thin ckpt vs thick ckpt; whether resume + replay reaches masters.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_checkpoint_thin
mkdir -p $OUTDIR

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"

cat > $OUTDIR/probe.sub <<EOF
universe = vanilla
executable = /bin/bash
arguments = $BASE/scripts/eval/probe_84_delta_checkpoint_thin_wrap.sh
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

cat > $BASE/scripts/eval/probe_84_delta_checkpoint_thin_wrap.sh <<'WRAP'
#!/bin/bash
set -e
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_84_delta_checkpoint_thin
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INTEGRAL_STR="-1,2,1,0,1,2,1,1,-3,0,0"
CKPT=$OUTDIR/beam.ckpt

export PYTHONUNBUFFERED=1

echo "===== STAGE 1: thin checkpoint, --max_steps 100 ====="
$PYTHON -u $BASE/scripts/eval/delta_onestep_worker.py \
  --topology $TOPOLOGY --integral="$INTEGRAL_STR" \
  --output $OUTDIR/stage1_result.pkl \
  --model-checkpoint $MODEL --beam_width 40 --max_steps 100 \
  --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 1 \
  --checkpoint $CKPT --checkpoint-mode thin \
  --checkpoint-interval 25 --checkpoint-time-seconds 60 \
  2>&1 | tee $OUTDIR/stage1.log

echo
echo "===== Checkpoint files after stage 1 ====="
ls -la $CKPT* 2>&1 || true
echo
echo "===== STAGE 2: --resume (replay) to completion ====="
$PYTHON -u $BASE/scripts/eval/delta_onestep_worker.py \
  --topology $TOPOLOGY --integral="$INTEGRAL_STR" \
  --output $OUTDIR/stage2_result.pkl \
  --model-checkpoint $MODEL --beam_width 40 --max_steps 5000 \
  --prime 1009 --device cpu -v --no-paper-masters-only --n_threads 1 \
  --checkpoint $CKPT --checkpoint-mode thin --resume \
  --checkpoint-interval 50 --checkpoint-time-seconds 300 \
  2>&1 | tee $OUTDIR/stage2.log

echo
echo "===== DONE ====="
WRAP
chmod +x $BASE/scripts/eval/probe_84_delta_checkpoint_thin_wrap.sh

condor_submit $OUTDIR/probe.sub
