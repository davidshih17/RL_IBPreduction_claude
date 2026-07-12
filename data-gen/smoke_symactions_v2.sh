#!/bin/bash
# Smoke v2 (after the sym-elim=seed fix) + CONTROL run without --sym-actions at the
# same seeds/sectors, to separate pre-existing failures from symmetry-induced ones.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
COMMON="--topology $BASE/topology_input/pentagonbox --n_scrambles 300 --min_steps 5 \
        --max_steps 20 --prime 1009 --start_seed 424242 \
        --restrict-sectors-file $BASE/results/canonical_sectors_tkey.txt"
echo "=== WITH --sym-actions ==="
$PY -u $BASE/data-gen/generate_multisector_data.py $COMMON --sym-actions \
    --output $BASE/data-gen/sectortest/smoke_symactions_v2.jsonl 2>&1 | tail -4
echo ""
echo "=== CONTROL (no sym actions) ==="
$PY -u $BASE/data-gen/generate_multisector_data.py $COMMON \
    --output $BASE/data-gen/sectortest/smoke_control_v2.jsonl 2>&1 | tail -4
echo ""
echo "=== audit (sym run) ==="
$PY $BASE/analysis/audit_symaction_samples.py $BASE/data-gen/sectortest/smoke_symactions_v2.jsonl 2>&1
