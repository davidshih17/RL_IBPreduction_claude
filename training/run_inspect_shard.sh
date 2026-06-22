#!/bin/bash
# Inspect one shard to measure in-RAM cost (esp. subs_raw Python lists)
set -e
cd /home/shih/work/SAILIR_phase2
mkdir -p logs
PYTHONUNBUFFERED=1 python -u training/inspect_shard.py \
    data/pentagonbox_10x_packed/shard_0/train.pt \
    2>&1 | tee logs/inspect_shard_0_train.log
