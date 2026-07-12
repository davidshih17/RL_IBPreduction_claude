#!/bin/bash
# Gate the staged symmetry router:
#  1. build + gate the per-sector composite canonicalization maps
#  2. standalone staged_rule validation (descent, step-1 sectors, fixpoint survivors)
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
echo "=== 1. sector_canon_maps build + gate ==="
$PY reduction/sector_canon_maps.py 2>&1
echo ""
echo "=== 2. staged_rule validation ==="
SAILIR_SECTOR_RANK=1 $PY analysis/validate_staged_router.py 2>&1
