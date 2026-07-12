#!/usr/bin/env python
"""How much real worker traffic lands in the sectors that have a nontrivial pure
within-sector group? Scan dispatched targets of the sectorrank + poststrip runs."""
import os, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
PURE = set(pickle.load(open(os.path.join(BASE, "results/pure_within_stabilizers.pkl"), "rb")))


def sector_of(t):
    return sum(1 << k for k in range(8) if t[k] > 0)


for arm in ("gate_small_sectorrank/design1", "m1_sectorrank/design1",
            "m2_sectorrank/design1", "m3_sectorrank/design1",
            "m1_poststrip/design1", "m2_poststrip/design1", "m3_poststrip/design1"):
    d = os.path.join(BASE, "results/ab_symmetry", arm, "work/results")
    if not os.path.isdir(d):
        continue
    tot = 0; hit = 0
    for fn in os.listdir(d):
        if fn.endswith(".pkl"):
            r = pickle.load(open(os.path.join(d, fn), "rb"))
            tot += 1
            if sector_of(tuple(r["original_integral"])) in PURE:
                hit += 1
    print(f"{arm:<28} dispatched={tot:<5} in pure-group sectors: {hit} "
          f"({100*hit/max(1,tot):.1f}%)")
