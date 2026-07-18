#!/bin/bash
# Rebuild the GR routing chain under the empirically-locked CLEAN-DEN convention
# (den rows must be exactly +1): provider gate, orbit comparison, composite
# maps, canonical masters, router smoke, FIRE-oracle zero cross-check.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE/reduction
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
rm -f $BASE/results/sector_canon_maps_GR.pkl
{
  echo "=== 1. canonicalize_GR gate (clean-den) ==="
  PYTHONUNBUFFERED=1 $PY canonicalize_GR.py
  echo
  echo "=== 2. canonical sectors: clean-only orbit rebuild vs pkl ==="
  PYTHONUNBUFFERED=1 $PY verify_canonical_sectors_GR.py
  echo
  echo "=== 3. sector_canon_maps rebuild + gate (clean edges) ==="
  PYTHONUNBUFFERED=1 $PY sector_canon_maps.py
  echo
  echo "=== 4. canonical masters rebuild + gate ==="
  PYTHONUNBUFFERED=1 $PY canonical_masters.py
  echo
  echo "=== 5. router smoke on 27 benchmark targets ==="
  PYTHONUNBUFFERED=1 $PY smoke_gr_router.py
  echo
  echo "=== 6. router vs FIRE-oracle zero cross-check ==="
  PYTHONUNBUFFERED=1 $PY smoke_gr_router_vs_oracle.py
} > logs/clean_den_rebuild_v2.log 2>&1
