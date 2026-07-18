#!/bin/bash
# BEHAVIORAL gate of the general engine (canonicalize2 + gravity3L_transforms_v2)
# against the production GR chain: transform-set identity is unattainable (the
# general engine legitimately finds MORE valid symmetries — 8 external variants
# vs 4 hand-coded), so the gate is downstream equivalence:
#   1. canonical sector orbits == canonical_sectors_GR_v2 (298, same reps)
#   2. composite canon maps: full reachability (725/725), corner-exact
#   3. canonical masters: 45 true masters, 23 merges, same dictionary shape
#   4. router smoke on the 27 benchmark targets: 0 structural fails
#   5. FIRE-oracle zero cross-check: 0 false zeros
# Runs with SAILIR_CANONICALIZE=canonicalize2 and a scratch canon-maps pkl so
# production artifacts are untouched.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE/reduction
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
export SAILIR_CANONICALIZE=canonicalize2
export SAILIR_CANON_MAPS_PKL=$BASE/results/sector_canon_maps_GR_engine2.pkl
rm -f $SAILIR_CANON_MAPS_PKL
{
  echo "=== 1. canonical sector orbits (canonicalize2) vs v2 pkl ==="
  PYTHONUNBUFFERED=1 $PY verify_canonical_sectors_GR2.py
  echo
  echo "=== 2. sector_canon_maps rebuild + gate (canonicalize2) ==="
  PYTHONUNBUFFERED=1 $PY sector_canon_maps.py
  echo
  echo "=== 3. canonical masters rebuild + gate (canonicalize2) ==="
  PYTHONUNBUFFERED=1 $PY canonical_masters.py
  echo
  echo "=== 4. router smoke on 27 benchmark targets (canonicalize2) ==="
  PYTHONUNBUFFERED=1 $PY smoke_gr_router.py
  echo
  echo "=== 5. router vs FIRE-oracle zero cross-check (canonicalize2) ==="
  PYTHONUNBUFFERED=1 $PY smoke_gr_router_vs_oracle.py
} > logs/engine2_behavioral_gate_v1.log 2>&1
