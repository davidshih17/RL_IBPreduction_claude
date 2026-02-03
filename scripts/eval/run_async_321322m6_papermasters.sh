#!/bin/bash
# Async parallel reduction on I[3,2,1,3,2,2,-6] with paper masters only and straggler handling

cd /het/p4/dshih/jet_images-deep_learning/RL_IBPreduction_claude

PYTHON=/het/p4/dshih/conda_envs/rl_dilogs/bin/python

$PYTHON -u scripts/eval/hierarchical_reduction_async.py \
    --integral 3,2,1,3,2,2,-6 \
    --output results/reduction_321322m6_async_papermasters.pkl \
    --work-dir /het/p4/dshih/scratch/ibp_async_321322m6_papermasters \
    --beam_width 20 \
    --prime 1009 \
    --check-interval 5 \
    --paper-masters-only \
    --straggler-timeout 1800 \
    --straggler-cpus 8
