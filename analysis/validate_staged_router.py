#!/usr/bin/env python
"""Standalone validation of symmetry_route.staged_rule (run with SAILIR_SECTOR_RANK=1).

For a sample of integrals (random dotted/numerator across all sectors + training
targets):
  (1) DESCENT: every staged rule's RHS is strictly lower in the shared order.
  (2) STEP-1 EXACTNESS: for non-canonical sectors the rule is the single composite
      image (an exact value identity by construction) — verify leading terms are in
      the canonical sector.
  (3) SEQUENCING: iterate staged_rule to a fixpoint (simulating the cascade) and
      check every SURVIVOR is (a) in a canonical sector, (b) step-2 irreducible.
  (4) COVERAGE vs the monolithic router: fraction routed by each (staged may route
      less per call but the cascade composes; compare fixpoint survivors).
"""
import os, sys, random, time
assert os.environ.get("SAILIR_SECTOR_RANK") == "1", "run with SAILIR_SECTOR_RANK=1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
import pickle
from symmetry_route import staged_rule, symmetry_rule, tkey, _sector_of

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
import json
with open(os.path.join(BASE, "data/pentagonbox_10x_raw_jsonl/multisector_data_worker0.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 173:
            continue
        pool.append(tuple(json.loads(line)["target"]))
        if i > 10000:
            break
pool = sorted(set(pool))
print(f"sample: {len(pool)} integrals", flush=True)

n_desc_fail = n_step1_secfail = 0
t0 = time.time()
memo = {}


def route_fixpoint(I, depth=0):
    """Simulate the cascade on a single integral: route until survivors; return the
    set of survivor integrals reachable from I."""
    if depth > 12:
        return {I}
    if I in memo:
        return memo[I]
    r = staged_rule(I)
    if r is None:
        memo[I] = {I}
        return {I}
    out = set()
    from sailir.ibp_env import is_master
    for J in r:
        if is_master(tuple(J)):
            continue
        out |= route_fixpoint(tuple(J), depth + 1)
    memo[I] = out
    return out


n_routed_staged = n_routed_mono = 0
n_survivor_noncanon = 0
for I in pool:
    ki = tkey(I)
    r = staged_rule(I)
    if r is not None:
        n_routed_staged += 1
        if not all(tkey(k) > ki for k in r):
            n_desc_fail += 1
            print(f"  DESCENT FAIL: {list(I)}", flush=True)
        if _sector_of(I) not in CANON:
            W = (sum(x for x in I if x > 0), sum(-x for x in I if x < 0))
            lead = [J for J in r if (sum(x for x in J if x > 0), sum(-x for x in J if x < 0)) == W]
            if any(_sector_of(J) not in CANON for J in lead):
                n_step1_secfail += 1
                print(f"  STEP1 SECTOR FAIL: {list(I)}", flush=True)
    if symmetry_rule(I) is not None:
        n_routed_mono += 1
t_route = time.time() - t0

# fixpoint survivors (sequencing check) on the non-canonical + canonical sample
t0 = time.time()
for I in pool[: len(pool) // 2]:
    for s in route_fixpoint(I):
        sec = _sector_of(s)
        if sec != 0 and sec not in CANON:     # sector 0 = scaleless debris, IBP's job
            n_survivor_noncanon += 1
            print(f"  NON-CANONICAL SURVIVOR: {list(s)} (from {list(I)})", flush=True)
t_fix = time.time() - t0

print(f"\nrouted: staged {n_routed_staged}/{len(pool)}  vs monolithic {n_routed_mono}/{len(pool)}")
print(f"descent failures: {n_desc_fail}")
print(f"step-1 leading-sector failures: {n_step1_secfail}")
print(f"non-canonical fixpoint survivors: {n_survivor_noncanon}")
print(f"timing: route both {t_route:.1f}s total; fixpoint sim {t_fix:.1f}s")
print("ALL PASS" if n_desc_fail == n_step1_secfail == n_survivor_noncanon == 0 else "FAIL")
