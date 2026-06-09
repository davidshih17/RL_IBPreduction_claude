#!/bin/bash
# Regression test: generate trianglebox training data with the same fixed seed
# using (a) the un-refactored SAILIR (main repo) and (b) the refactored
# SAILIR_phase2, and compare the JSONL outputs byte-by-byte.
set -u
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUTDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/symmetries/tmp_regression
mkdir -p $OUTDIR/old $OUTDIR/new
LOG=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/symmetries/logs/trianglebox_regression.log
: > $LOG

SEED=12345
SCRAMBLES=20

echo "==== UN-REFACTORED (SAILIR main) ====" >> $LOG
cd /het/p4/dshih/jet_images-deep_learning/SAILIR
PYTHONUNBUFFERED=1 "$PY" -u scripts/data_gen/generate_multisector_data.py \
    --n_scrambles $SCRAMBLES \
    --min_steps 3 --max_steps 5 \
    --output $OUTDIR/old/trianglebox.jsonl \
    --start_seed $SEED \
    --prime 1009 \
    --ibp_path scripts/data_gen/IBP \
    --li_path scripts/data_gen/LI \
    >> $LOG 2>&1
echo "old exit=$?  $(wc -l < $OUTDIR/old/trianglebox.jsonl 2>/dev/null) samples" >> $LOG

echo >> $LOG
echo "==== REFACTORED (SAILIR_phase2, --topology) ====" >> $LOG
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHONUNBUFFERED=1 "$PY" -u scripts/data_gen/generate_multisector_data.py \
    --topology topology_input/trianglebox \
    --n_scrambles $SCRAMBLES \
    --min_steps 3 --max_steps 5 \
    --output $OUTDIR/new/trianglebox.jsonl \
    --start_seed $SEED \
    --prime 1009 \
    --ibp_path topology_input/trianglebox/IBP \
    --li_path topology_input/trianglebox/LI \
    >> $LOG 2>&1
echo "new exit=$?  $(wc -l < $OUTDIR/new/trianglebox.jsonl 2>/dev/null) samples" >> $LOG

echo >> $LOG
echo "==== diff ====" >> $LOG
if diff -q $OUTDIR/old/trianglebox.jsonl $OUTDIR/new/trianglebox.jsonl >> $LOG 2>&1; then
    echo "IDENTICAL outputs" >> $LOG
else
    echo "DIFFERENT outputs" >> $LOG
    diff $OUTDIR/old/trianglebox.jsonl $OUTDIR/new/trianglebox.jsonl | head -50 >> $LOG
fi

tail -25 $LOG
