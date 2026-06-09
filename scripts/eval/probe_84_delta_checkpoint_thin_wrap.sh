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
