#!/bin/bash
# Smoke test: run generate_multisector_data.py on both topologies
# with --n_scrambles 5 (tiny). Each output JSONL is appended to logs/.
set -u
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
mkdir -p scripts/symmetries/logs scripts/symmetries/tmp_datagen
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

LOG=scripts/symmetries/logs/datagen_smoke.log
echo "==== trianglebox ====" > $LOG
PYTHONUNBUFFERED=1 "$PY" -u data-gen/generate_multisector_data.py \
    --topology topology_input/trianglebox \
    --n_scrambles 5 \
    --min_steps 2 --max_steps 4 \
    --output scripts/symmetries/tmp_datagen/tb_smoke.jsonl \
    --start_seed 0 \
    >> $LOG 2>&1
echo "tb exit=$?  size=$(wc -l < scripts/symmetries/tmp_datagen/tb_smoke.jsonl 2>/dev/null) samples" >> $LOG

echo "==== pentagon-box ====" >> $LOG
PYTHONUNBUFFERED=1 "$PY" -u data-gen/generate_multisector_data.py \
    --topology topology_input/pentagonbox \
    --n_scrambles 5 \
    --min_steps 2 --max_steps 4 \
    --output scripts/symmetries/tmp_datagen/pb_smoke.jsonl \
    --start_seed 0 \
    >> $LOG 2>&1
echo "pb exit=$?  size=$(wc -l < scripts/symmetries/tmp_datagen/pb_smoke.jsonl 2>/dev/null) samples" >> $LOG

cat $LOG
