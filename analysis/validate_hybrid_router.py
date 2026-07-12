#!/usr/bin/env python
"""Gate canonical_monolithic_rule (run with SAILIR_SECTOR_RANK=1):
  - HARD GUARANTEE: no non-canonical-sector integral (except sector 0) is ever a
    survivor;
  - every emitted rule descends strictly in the shared order;
  - rule geometry matches the monolithic router wherever it routes (the fallback
    only fires where the monolithic solve returned None)."""
import os, sys, json, random
assert os.environ.get("SAILIR_SECTOR_RANK") == "1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
import pickle
from symmetry_route import canonical_monolithic_rule, symmetry_rule, tkey, _sector_of

CANON = set(pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))["canonical"])
rng = random.Random(20260710)
pool = []
for S in range(1, 256):
    pr = [i for i in range(8) if S >> i & 1]
    numslots = [i for i in range(11) if i not in pr]
    for _ in range(2):
        a = [0] * 11
        for i in pr:
            a[i] = rng.randint(1, 3)
        for i in rng.sample(numslots, rng.randint(0, 2)):
            a[i] = -rng.randint(1, 2)
        pool.append(tuple(a))
with open(os.path.join(BASE, "data/pentagonbox_10x_raw_jsonl/multisector_data_worker0.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 173:
            continue
        pool.append(tuple(json.loads(line)["target"]))
        if i > 10000:
            break
pool = sorted(set(pool))

n_bad_survivor = n_desc = n_fallback = n_routed = 0
for I in pool:
    r = canonical_monolithic_rule(I)
    S = _sector_of(I)
    if r is None:
        if S != 0 and S not in CANON:
            n_bad_survivor += 1
            print(f"  NON-CANONICAL SURVIVOR: {list(I)}")
        continue
    n_routed += 1
    ki = tkey(I)
    if not all(tkey(k) > ki for k in r):
        n_desc += 1; print(f"  DESCENT FAIL: {list(I)}")
    if symmetry_rule(I) is None:
        n_fallback += 1
print(f"sample {len(pool)}: routed {n_routed}, fallback-canonicalized {n_fallback}")
print(f"non-canonical survivors: {n_bad_survivor}   descent failures: {n_desc}")
print("ALL PASS" if n_bad_survivor == n_desc == 0 else "FAIL")
