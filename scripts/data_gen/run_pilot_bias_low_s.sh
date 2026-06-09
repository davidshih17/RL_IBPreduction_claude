#!/bin/bash
# Pilot study: compare baseline vs --bias-low-s-elim on sector 254 (L7 D1-off).
# 1000 scrambles each. Output JSONL files for analysis.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUT=$BASE/data/pilot_bias_low_s
mkdir -p $OUT

PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

cd $BASE
export PYTHONUNBUFFERED=1

echo "=== Pilot A: baseline (uniform elim) ===" | tee $OUT/run.log
$PY -u scripts/data_gen/generate_multisector_data.py \
    --topology topology_input/pentagonbox \
    --n_scrambles 1000 \
    --min_steps 5 --max_steps 20 \
    --restrict-sectors 254 \
    --output $OUT/pilot_A_baseline.jsonl \
    --start_seed 0 \
    --prime 1009 2>&1 | tee -a $OUT/run.log

echo | tee -a $OUT/run.log
echo "=== Pilot B: bias-low-s-elim ===" | tee -a $OUT/run.log
$PY -u scripts/data_gen/generate_multisector_data.py \
    --topology topology_input/pentagonbox \
    --n_scrambles 1000 \
    --min_steps 5 --max_steps 20 \
    --restrict-sectors 254 \
    --bias-low-s-elim \
    --output $OUT/pilot_B_biased.jsonl \
    --start_seed 1000000 \
    --prime 1009 2>&1 | tee -a $OUT/run.log

echo | tee -a $OUT/run.log
echo "=== Both pilots done ===" | tee -a $OUT/run.log
wc -l $OUT/pilot_*.jsonl | tee -a $OUT/run.log
