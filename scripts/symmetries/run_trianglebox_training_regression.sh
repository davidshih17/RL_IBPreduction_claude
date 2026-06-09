#!/bin/bash
# Trianglebox training pipeline regression:
#   1. Preprocess the (already bit-identical) regression jsonl with both old
#      and new code, diff the resulting .pt files.
#   2. Run train_classifier.py for 3 epochs with both old and new code
#      using identical random seed and config. Compare loss/accuracy.
set -u
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
ROOT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
TMP=$ROOT/scripts/symmetries/tmp_regression
LOG=$ROOT/scripts/symmetries/logs/trianglebox_training_regression.log
: > $LOG

SEED=12345

# ---- Step 1: Preprocess the regression JSONL with old + new code ----
# Use the already-existing $TMP/new/trianglebox.jsonl as input for both
# (we know it's bit-identical to $TMP/old/trianglebox.jsonl).
mkdir -p $TMP/packed_old $TMP/packed_new

echo "==== Preprocess (OLD) ====" >> $LOG
cd /het/p4/dshih/jet_images-deep_learning/SAILIR
PYTHONUNBUFFERED=1 "$PY" -u scripts/data_gen/preprocess_to_tensors.py \
    --input $TMP/new/trianglebox.jsonl \
    --output_dir $TMP/packed_old \
    --val_split 0.2 --test_split 0.2 \
    --seed $SEED >> $LOG 2>&1
echo "old preprocess exit=$?" >> $LOG

echo >> $LOG
echo "==== Preprocess (NEW) ====" >> $LOG
cd $ROOT
PYTHONUNBUFFERED=1 "$PY" -u scripts/data_gen/preprocess_to_tensors.py \
    --topology topology_input/trianglebox \
    --input $TMP/new/trianglebox.jsonl \
    --output_dir $TMP/packed_new \
    --val_split 0.2 --test_split 0.2 \
    --seed $SEED >> $LOG 2>&1
echo "new preprocess exit=$?" >> $LOG

# Diff .pt files (binary - if identical the refactor preserved preprocess)
echo >> $LOG
echo "==== diff .pt files ====" >> $LOG
for split in train val test; do
    cmp $TMP/packed_old/${split}.pt $TMP/packed_new/${split}.pt && \
        echo "  $split.pt: BYTE-IDENTICAL" >> $LOG || \
        echo "  $split.pt: DIFFERS (size: $(stat -c '%s' $TMP/packed_old/${split}.pt) vs $(stat -c '%s' $TMP/packed_new/${split}.pt))" >> $LOG
done

# ---- Step 2: Train 3 epochs with both ----
mkdir -p $TMP/ckpt_old $TMP/ckpt_new

echo >> $LOG
echo "==== Train (OLD), 3 epochs, CPU, tiny model ====" >> $LOG
cd /het/p4/dshih/jet_images-deep_learning/SAILIR
PYTHONUNBUFFERED=1 "$PY" -u scripts/train/train_classifier.py \
    --data_dir $TMP/packed_old \
    --output_dir $TMP/ckpt_old \
    --epochs 3 --batch_size 4 --num_workers 0 \
    --embed_dim 32 --n_heads 2 --n_expr_layers 1 --n_cross_layers 1 --n_subs_layers 1 \
    --device cpu --prime 1009 \
    >> $LOG 2>&1
echo "old train exit=$?" >> $LOG

echo >> $LOG
echo "==== Train (NEW), 3 epochs, CPU, tiny model ====" >> $LOG
cd $ROOT
PYTHONUNBUFFERED=1 "$PY" -u scripts/train/train_classifier.py \
    --topology topology_input/trianglebox \
    --data_dir $TMP/packed_new \
    --output_dir $TMP/ckpt_new \
    --epochs 3 --batch_size 4 --num_workers 0 \
    --embed_dim 32 --n_heads 2 --n_expr_layers 1 --n_cross_layers 1 --n_subs_layers 1 \
    --device cpu --prime 1009 \
    >> $LOG 2>&1
echo "new train exit=$?" >> $LOG

# Extract loss lines for comparison
echo >> $LOG
echo "==== Old loss curve (Epoch summaries) ====" >> $LOG
grep -E 'Epoch [0-9]+:.*loss=|val_loss=|Val Loss=|Loss:|train: top1' $LOG | head -30 >> $LOG

tail -80 $LOG
