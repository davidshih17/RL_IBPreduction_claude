#!/bin/bash
# Certify the definitive canonical-sector list, then verify the pipeline claim:
#  1. rebuild results/canonical_sectors_tkey.pkl from the fixed canonical_rep
#  2. run the verify_canonical_rep.py gate (must be ALL PASS)
#  3. run verify_canonical_dispatch.py (dispatched-target sectors in m1/m2/m3
#     symmetry arms vs the canonical list + symmetry_rule stress test on
#     non-canonical dotted/numerator integrals)
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
echo "=== 1. rebuild canonical sector map ==="
$PY reduction/build_canonical_sectors_tkey.py 2>&1
echo ""
echo "=== 2. verify_canonical_rep gate ==="
$PY reduction/verify_canonical_rep.py 2>&1
echo ""
echo "=== 3. dispatch + stress verification ==="
$PY analysis/verify_canonical_dispatch.py 2>&1
