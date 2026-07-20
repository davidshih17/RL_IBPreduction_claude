#!/usr/bin/env python
"""Validate banked worker one-step results against the sector-senior order
contract: every term of final_expr must be STRICTLY BELOW the start integral
(tkey(term) > tkey(start); smaller tkey = higher in the order — the eliminator
pivots on min-tkey). Violating pkls are moved to <dir>/../results_quarantine/.

Usage: SAILIR_TOPOLOGY=... SAILIR_SECTOR_RANK=1 validate_banked_results.py
           <work/results dir> [--apply]
Without --apply: report only.
"""
import os, sys, glob, pickle, shutil
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_route import tkey

d = os.path.abspath(sys.argv[1])
apply_mode = "--apply" in sys.argv
qdir = os.path.join(os.path.dirname(d), "results_quarantine")

n_ok = n_fail = n_viol = 0
viol_files = []
for f in sorted(glob.glob(d + "/*.pkl")):
    try:
        r = pickle.load(open(f, "rb"))
    except Exception:
        n_fail += 1
        continue
    start = r.get("original_integral") or r.get("integral")
    fe = r.get("final_expr")
    if start is None or fe is None or not r.get("success", True):
        n_fail += 1
        continue
    k0 = tkey(tuple(start))
    bad = [t for t in fe if tkey(tuple(t)) <= k0]
    if bad:
        n_viol += 1
        viol_files.append(f)
        print(f"VIOLATION {os.path.basename(f)}")
        print(f"  start {list(start)}")
        for t in bad[:4]:
            print(f"  term-not-below: {list(t)}")
    else:
        n_ok += 1

print(f"\n{d}: ok={n_ok} violations={n_viol} unusable/failed={n_fail}")
if apply_mode and viol_files:
    os.makedirs(qdir, exist_ok=True)
    for f in viol_files:
        shutil.move(f, os.path.join(qdir, os.path.basename(f)))
    print(f"moved {len(viol_files)} pkls -> {qdir}")
elif viol_files:
    print("(report only; rerun with --apply to quarantine)")
