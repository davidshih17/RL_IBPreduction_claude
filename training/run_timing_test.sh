#!/bin/bash
set -euo pipefail
cd /home/shih/work/SAILIR_phase2

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1

python -u training/timing_test_sharded.py 2>&1 | tee logs/timing_test.log
