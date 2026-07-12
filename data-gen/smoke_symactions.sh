#!/bin/bash
# Smoke test of the production data-gen design (2026-07-11): canonical sectors only
# + within-sector symmetry relations in the action space.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT=$BASE/data-gen/sectortest/smoke_symactions.jsonl
$PY -u $BASE/data-gen/generate_multisector_data.py \
    --topology $BASE/topology_input/pentagonbox \
    --n_scrambles 300 --min_steps 5 --max_steps 20 \
    --prime 1009 --start_seed 424242 \
    --restrict-sectors-file $BASE/results/canonical_sectors_tkey.txt \
    --sym-actions \
    --output $OUT 2>&1
echo ""
echo "=== sample audit ==="
$PY $BASE/analysis/audit_symaction_samples.py $OUT 2>&1
