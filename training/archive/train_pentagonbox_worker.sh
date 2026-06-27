#!/bin/bash
# Pentagon-box training worker (Condor on GPU node). v2: batch=128 to fit
# in 16 GB / GPU after the v1 OOM at batch=256.
set -u
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT_DIR=$SAILIR_DIR/checkpoints/pentagonbox_v2
LOG_DIR=$OUT_DIR/logs
mkdir -p $LOG_DIR

cd $SAILIR_DIR
{
  echo "==== train_pentagonbox_worker.sh (v2, batch=128) ===="
  date
  echo "host: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1
  echo "===================================================="
} | tee -a $LOG_DIR/training.log

PYTHONUNBUFFERED=1 $PY -u training/train_classifier.py \
    --topology topology_input/pentagonbox \
    --data_dir data/pentagonbox_packed \
    --output_dir $OUT_DIR \
    --epochs 30 --batch_size 128 --lr 4e-4 \
    --embed_dim 256 --n_heads 4 \
    --n_expr_layers 2 --n_cross_layers 2 --n_subs_layers 2 \
    --prime 1009 --device cuda --num_workers 4 \
    2>&1 | tee -a $LOG_DIR/training.log

echo "training complete (exit=${PIPESTATUS[0]})" | tee -a $LOG_DIR/training.log
