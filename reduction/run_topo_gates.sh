#!/bin/bash
# Gate battery for the topology-keyed generalization.
# Part 1: PENTAGONBOX REGRESSION (SAILIR_TOPOLOGY unset -> default) — every
#         module gate must pass with numbers identical to the pre-refactor runs.
# Part 2: GRAVITY3L builds + gates (sector rank, composite canon maps,
#         canonical masters with the FIRE-basis dictionary).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE/reduction
{
  echo "############ PART 1: pentagonbox regression (default env) ############"
  echo "=== sector_rank gate ==="
  PYTHONUNBUFFERED=1 $PY sector_rank.py
  echo
  echo "=== canonical_masters gate (pentagonbox) ==="
  PYTHONUNBUFFERED=1 $PY canonical_masters.py
  echo
  echo "=== symmetry_route demo (pentagonbox, monolithic) ==="
  PYTHONUNBUFFERED=1 $PY symmetry_route.py
  echo
  echo "############ PART 2: gravity3L builds ############"
  export SAILIR_TOPOLOGY=gravity3L
  export SAILIR_SECTOR_RANK=1
  echo "=== sector_rank gate (gravity3L) ==="
  PYTHONUNBUFFERED=1 $PY sector_rank.py
  echo
  echo "=== sector_canon_maps build + gate (gravity3L) ==="
  PYTHONUNBUFFERED=1 $PY sector_canon_maps.py
  echo
  echo "=== canonical_masters build + gate (gravity3L / FIRE-68) ==="
  PYTHONUNBUFFERED=1 $PY canonical_masters.py
} > logs/topo_gates_v1.log 2>&1
