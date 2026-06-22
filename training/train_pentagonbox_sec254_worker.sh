#!/bin/bash
# Sector-254 dedicated classifier training.
# Train from scratch on sec254-only data (~500k samples expected). Smaller
# distribution → faster convergence; we use the same architecture as v2 but
# expect fewer epochs to suffice.
set -u
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT_DIR=$SAILIR_DIR/checkpoints/pentagonbox_sec254_v1
LOG_DIR=$OUT_DIR/logs
mkdir -p $LOG_DIR

cd $SAILIR_DIR
{
  echo "==== train_pentagonbox_sec254_worker.sh ===="
  date
  echo "host: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1
  echo "==========================================="
} | tee -a $LOG_DIR/training.log

PYTHONUNBUFFERED=1 $PY -u training/train_classifier.py \
    --topology topology_input/pentagonbox \
    --data_dir data/pentagonbox_sec254_packed \
    --output_dir $OUT_DIR \
    --epochs 30 --batch_size 128 --lr 4e-4 \
    --embed_dim 256 --n_heads 4 \
    --n_expr_layers 2 --n_cross_layers 2 --n_subs_layers 2 \
    --prime 1009 --device cuda --num_workers 4 \
    2>&1 | tee -a $LOG_DIR/training.log

echo "training complete (exit=${PIPESTATUS[0]})" | tee -a $LOG_DIR/training.log
