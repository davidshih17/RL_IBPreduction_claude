#!/bin/bash
# GR (3-loop gravity) data-gen smoke: 300 scrambles restricted to the 292 canonical
# sectors, FIRE-68 masters + corner fallback. The per-scramble coefficient gate is
# the empirical basis check: failures concentrated in the orbit-paired master
# sectors (885/891/767/1013/1019) would indicate per-sector basis incompleteness.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT=$BASE/data-gen/sectortest/smoke_gravity3L.jsonl
$PY -u $BASE/data-gen/generate_multisector_data.py \
    --topology $BASE/topology_input/gravity3L \
    --n_scrambles 300 --min_steps 5 --max_steps 20 \
    --prime 1009 --start_seed 777000 \
    --restrict-sectors-file $BASE/results/canonical_sectors_GR.txt \
    --output $OUT 2>&1
echo ""
echo "=== outcome by sector (success samples vs failures) ==="
$PY - "$OUT" <<'EOF'
import json, sys, collections
ok = collections.Counter()
for line in open(sys.argv[1]):
    ok[json.loads(line)["sector_id"]] += 1
print(f"sectors with samples: {len(ok)}; total samples: {sum(ok.values())}")
print("top sectors:", ok.most_common(8))
EOF
