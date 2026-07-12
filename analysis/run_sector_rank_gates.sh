#!/bin/bash
# Gate the sector-senior order wiring:
#  1. sector_rank.py self-gate (contract clauses)
#  2. verify_canonical_rep.py with SAILIR_SECTOR_RANK=1 (rep must remain the
#     survivor under the NEW order; symmetry_route + canonical_rep both flagged)
#  3. import smoke test of beam_search_v7 with the flag (RANK table loads, keys work)
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
echo "=== 1. sector_rank self-gate ==="
$PY reduction/sector_rank.py 2>&1
echo ""
echo "=== 2. verify_canonical_rep with SAILIR_SECTOR_RANK=1 ==="
SAILIR_SECTOR_RANK=1 $PY reduction/verify_canonical_rep.py 2>&1
echo ""
echo "=== 3. beam_search_v7 key smoke test with SAILIR_SECTOR_RANK=1 ==="
SAILIR_SECTOR_RANK=1 $PY analysis/smoke_sector_rank_keys.py 2>&1
