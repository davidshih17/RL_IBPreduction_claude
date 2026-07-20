#!/usr/bin/env python
"""Inspect the two g127 cache entries forming the substitution cycle:
print success, steps, term count, and every RHS term whose sector-senior
weight is >= the start integral's (contract violations)."""
import os, sys, glob, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_route import tkey

D = os.path.join(ROOT, "results/gr_reduce/g127/work/results")
A = (3, 2, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, -2, 0)
B = (2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, -1, 0)

for t in (A, B):
    name = "_".join(str(x) for x in t)
    hits = glob.glob(D + f"/*_{name}.pkl")
    print(f"\n=== {list(t)}  tkey={tkey(t)}")
    for f in hits:
        r = pickle.load(open(f, "rb"))
        fe = r.get("final_expr") or {}
        viol = [k for k in fe if tkey(tuple(k)) >= tkey(t)]
        print(f"  {os.path.basename(f)}  mtime={os.path.getmtime(f):.0f}")
        print(f"    success={r.get('success')} steps={r.get('steps')} "
              f"terms={len(fe)} weight-violations={len(viol)}")
        for k in viol[:6]:
            print(f"    VIOLATION: {list(k)}  tkey={tkey(tuple(k))}")
