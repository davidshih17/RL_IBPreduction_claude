#!/usr/bin/env python
"""Count dispatched worker targets in non-canonical sectors for given work dirs.
Usage: scan_dispatch_sectors.py <arm-dir-relative-to-results/ab_symmetry> [...]"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
CANON = set(pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))["canonical"])


def sector_of(t):
    return sum(1 << k for k in range(8) if t[k] > 0)


for arm in sys.argv[1:]:
    d = os.path.join(BASE, "results/ab_symmetry", arm, "work/results")
    tot = 0; bad = []
    for fn in os.listdir(d):
        if fn.endswith(".pkl"):
            r = pickle.load(open(os.path.join(d, fn), "rb"))
            I = tuple(r["original_integral"]); tot += 1
            s = sector_of(I)
            if s != 0 and s not in CANON:
                bad.append((list(I), s))
    print(f"{arm:<30} dispatched={tot:<5} non-canonical: {len(bad)}")
    for b in bad[:5]:
        print(f"    {b}")
