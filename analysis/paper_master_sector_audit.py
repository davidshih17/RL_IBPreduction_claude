#!/usr/bin/env python
"""Audit: do Kira's paper masters live in OUR canonical sectors?

Under the production convention (PAPER_MASTERS_ONLY=True):
  - re-check the stuck corner I[1,0,0,0,0,1,0,1,0,0,0] and its orbit-mate;
  - for EVERY paper master: its sector, its clean-orbit canonical sector (our
    convention), and whether they AGREE. Every disagreement is a sector orbit whose
    merged (symmetry-routed) corner is NOT a paper master and CANNOT be reduced by
    IBP within its cone -> a worker dispatched on it hangs forever. This is the
    long-standing blocker of the m1/m3 symmetry arms.
  - also check per-orbit uniqueness of the paper-master sector (needed if we re-pin
    canonical := Kira-preferred sector).
"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)
from sailir.ibp_env import is_master, MASTERS_SET
from canonical_rep import clean_orbit, canonical_rep

def sec(t): return sum(1 << k for k in range(8) if t[k] > 0)

I0 = (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0)
M152 = (0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0)
print(f"PAPER convention: is_master(I0 sec161) = {is_master(I0)}")
print(f"                  is_master(M  sec152) = {is_master(M152)}")

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))
REP_OF = cm["rep_of"]

print(f"\npaper masters: {len(MASTERS_SET)}")
mismatch = 0
orbit_master_secs = {}
for m in sorted(MASTERS_SET):
    s = sec(m)
    rep_sec = REP_OF.get(s, s)
    orbit_master_secs.setdefault(rep_sec, set()).add(s)
    tag = "OK " if s == rep_sec else "MISMATCH"
    if s != rep_sec:
        mismatch += 1
    print(f"  {tag}  master {list(m)}  sector {s:>3}  canonical-of-orbit {rep_sec:>3}")

print(f"\nmasters in NON-canonical sectors: {mismatch}/{len(MASTERS_SET)}")
multi = {r: s for r, s in orbit_master_secs.items() if len(s) > 1}
print(f"orbits whose paper masters span MULTIPLE sectors: {len(multi)}")
for r, s in multi.items():
    print(f"   orbit rep {r}: master sectors {sorted(s)}")
