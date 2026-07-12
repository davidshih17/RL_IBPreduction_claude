#!/usr/bin/env python
"""Does the current --use-symmetry pipeline dispatch workers ONLY in canonical sectors?

 (A) EVIDENCE FROM REAL RUNS: every dispatched target (async_*.pkl original_integral)
     in the m1/m2/m3 symmetry arms (design1) -- what sector is it in?  Compare against
     the baseline arms.  Violation = a design1 dispatched target in a NON-canonical
     sector.
 (B) STRESS TEST OF THE ROUTER: random dotted/numerator integrals in NON-canonical
     sectors -> symmetry_rule.  Routed away (rule returned) is what the user's design
     philosophy requires.  A SURVIVOR in a non-canonical sector is a counterexample to
     "the pipeline always canonicalizes the sector first" -- possible in principle
     because the router descends in kappa and the |a| tiebreak on dotted integrals can
     point across sectors differently than the corner tiebreak.
"""
import os, sys, pickle, random
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonicalize import sector_of
from symmetry_route import symmetry_rule

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))
CANON = set(cm["canonical"]); REP = cm["rep_of"]
print(f"canonical sectors: {len(CANON)}")

# ---------- (A) real dispatched targets ----------
def scan(arm):
    d = os.path.join(BASE, "results/ab_symmetry", arm, "work/results")
    if not os.path.isdir(d):
        return None
    tot = 0; bad = []
    for fn in os.listdir(d):
        if not fn.endswith(".pkl"):
            continue
        r = pickle.load(open(os.path.join(d, fn), "rb"))
        I = tuple(r["original_integral"]); tot += 1
        s = sector_of(I)
        if s not in CANON:
            bad.append((list(I), s, REP[s]))
    return tot, bad

print("\n=== (A) dispatched-target sectors in real runs ===")
for tag in ("m1_poststrip", "m2_poststrip", "m3_poststrip"):
    for arm in (f"{tag}/design1", f"{tag}/base", f"{tag}/baseline"):
        r = scan(arm)
        if r is None:
            continue
        tot, bad = r
        print(f"  {arm:<28} dispatched={tot:<5} in NON-canonical sectors: {len(bad)}")
        for b in bad[:5]:
            print(f"      violation: I={b[0]}  sector {b[1]} (rep {b[2]})")

# ---------- (B) router stress test on non-canonical sectors ----------
print("\n=== (B) symmetry_rule on random NON-canonical dotted/numerator integrals ===")
rng = random.Random(20260710)
noncanon = [m for m in range(1, 256) if m not in CANON]
n = 0; routed = 0; survivors = []
for m in noncanon:
    pr = [i for i in range(8) if m >> i & 1]
    numslots = [i for i in range(11) if i not in pr]
    for _ in range(4):
        a = [0] * 11
        for i in pr:
            a[i] = rng.randint(1, 3)
        for i in rng.sample(numslots, rng.randint(0, 2)):
            a[i] = -rng.randint(1, 2)
        a = tuple(a); n += 1
        rule = symmetry_rule(a)
        if rule is not None:
            routed += 1
        else:
            survivors.append((list(a), sector_of(a), REP[sector_of(a)]))
print(f"  non-canonical sectors: {len(noncanon)}, random integrals tested: {n}")
print(f"  routed away by symmetry_rule: {routed}")
print(f"  SURVIVORS in non-canonical sectors (counterexamples): {len(survivors)}")
for s in survivors[:12]:
    print(f"      survivor: I={s[0]}  sector {s[1]} (rep {s[2]})")
