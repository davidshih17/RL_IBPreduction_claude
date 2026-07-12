#!/usr/bin/env python
"""Diagnose the universally-stuck corner I[1,0,0,0,0,1,0,1,0,0,0] (D1 D6 D8, sector 161).

Questions:
 (1) is it a paper master? are any of its clean-orbit corners paper masters?
 (2) what does canonical_rep say (legacy vs sector-senior)? which sector is canonical?
 (3) does symmetry_rule route it (legacy and sector-senior)? why not?
 (4) do the m1/m3 BASELINE final masters contain it or an orbit-mate?
"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import is_master
from canonical_rep import clean_orbit, canonical_rep, tkey
from symmetry_route import symmetry_rule
import canonicalize as C

I0 = (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0)
def sec(t): return sum(1 << k for k in range(8) if t[k] > 0)

print(f"I0 = {list(I0)}  sector {sec(I0)}")
print(f"SAILIR_SECTOR_RANK = {os.environ.get('SAILIR_SECTOR_RANK', '0')}")
print(f"(1) is_master(I0) [paper]: {is_master(I0)}")
orb = clean_orbit(I0)
print(f"    clean orbit: {len(orb)} members")
for J in sorted(orb, key=tkey):
    print(f"      {list(J)}  sector {sec(J):>3}  master={is_master(tuple(J))}  tkey={tkey(J)}")
rep = canonical_rep(I0)
print(f"(2) canonical_rep(I0) = {list(rep)}  sector {sec(rep)}")
cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))
print(f"    canonical sectors of the orbit: {[s for s in sorted(set(sec(J) for J in orb)) if s in set(cm['canonical'])]}")
r = symmetry_rule(I0)
print(f"(3) symmetry_rule(I0): {'None (survivor)' if r is None else f'{len(r)} terms'}")

for tag, path in (("m1 baseline", "results/ab_symmetry/m1_6prop/baseline/reduction.pkl"),
                  ("m3 baseline", "results/ab_symmetry/m3_5prop_deg3/baseline/reduction.pkl")):
    p = os.path.join(BASE, path)
    if not os.path.exists(p):
        print(f"(4) {tag}: pkl missing at {path}"); continue
    fe = pickle.load(open(p, "rb"))
    fe = fe.get("final_expression", fe.get("final_expr", {}))
    hits = [tuple(k) for k in fe if tuple(k) in orb]
    print(f"(4) {tag}: {len(fe)} masters; orbit members among them: {[list(h) for h in hits]}")
