#!/usr/bin/env python
"""Classify every base symmetry record into the 2x2:
  axis A: pure (denominators -> single props AND each ISP row -> single slot, coeff +-1,
          no constant) vs affine (some ISP row is a genuine combination / has a constant)
  axis B: at the record's source sector: within-sector (present props permute among
          themselves) vs sector-changing (image prop set differs)
Counts per box + one example each.  Also flags: records whose den rows are not single
entries at their own source sector (not usable as clean rewrites there).
"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
P = 1009; N = 11

SYM, REL = None, None
rich = os.path.join(BASE, "analysis", "puresym_rich.pkl")
RICH = pickle.load(open(rich, "rb"))          # (source_sector, (M,c), loop_subst, Msym, csym)

# also classify the 17 relations (numeric only)
SYMREL = pickle.load(open(os.path.join(BASE, "analysis", "puresym_transforms.pkl"), "rb"))
_, RELS = SYMREL

def props(S): return [i for i in range(8) if S >> i & 1]

def classify(S, M, c):
    pr = props(S)
    img = []
    for i in pr:
        row = M.get(i, {})
        if len(row) != 1:
            return None                        # den row not a single prop at source sector
        (j, co), = row.items()
        if j >= 8:
            return None                        # den maps to an ISP slot
        img.append(j)
    within = sorted(img) == sorted(pr)
    pure = True
    for i in (8, 9, 10):
        row = M.get(i, {}); cc = c.get(i, 0) % P
        if cc or len(row) != 1:
            pure = False; break
        (j, co), = row.items()
        if co % P not in (1, P - 1):
            pure = False; break
    return ("pure" if pure else "affine", "within" if within else "changing")

boxes = {}
examples = {}
n_unusable = 0
for (S, (M, c), ls, Msym, csym) in RICH:
    cl = classify(S, M, c)
    if cl is None:
        n_unusable += 1; continue
    boxes[cl] = boxes.get(cl, 0) + 1
    if cl not in examples:
        examples[cl] = (S, ls, {i: Msym.get(i, {}) for i in (8, 9, 10)}, csym)

print("=== sectorSymmetries base records (108), classified at their source sector ===")
for cl in (("pure", "within"), ("pure", "changing"), ("affine", "within"), ("affine", "changing")):
    print(f"  {cl[0]:>6} x {cl[1]:<9}: {boxes.get(cl, 0)}")
print(f"  unusable-at-source (den row not a single prop): {n_unusable}")
for cl, (S, ls, isp, csym) in sorted(examples.items()):
    print(f"\n  EXAMPLE {cl}:  source sector {S} (props {[p+1 for p in props(S)]})")
    print(f"    loop_subst: {ls}")
    for i in (8, 9, 10):
        print(f"    D{i+1} row: {isp[i]}   const: {csym.get(i, '0')}")

print("\n=== sectorRelations records (17), numeric, classified at their source sector ===")
rboxes = {}; run = 0
for (S, (M, c)) in RELS:
    cl = classify(S, M, c)
    if cl is None:
        run += 1; continue
    rboxes[cl] = rboxes.get(cl, 0) + 1
for cl in (("pure", "within"), ("pure", "changing"), ("affine", "within"), ("affine", "changing")):
    print(f"  {cl[0]:>6} x {cl[1]:<9}: {rboxes.get(cl, 0)}")
print(f"  unusable-at-source: {run}")
