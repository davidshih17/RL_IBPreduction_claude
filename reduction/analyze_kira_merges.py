#!/usr/bin/env python
"""Which 4 FIRE masters did Kira's symmetry-enabled run merge away, and do they
correlate with our engine's map coefficients?

Hypothesis: Kira accepts a symmetry as a value identity ONLY when the true
momentum map preserves every denominator with coefficient +1 (no eikonal sign
flips). Prediction: the FIRE masters absent from Kira's 64-list are exactly
those whose canonical image has ALL-den-coefficient +1; the ones Kira kept
(despite an ing-applicable record) only have flip (-1) maps.

For each of the 24 moved FIRE masters, find ALL v4-store transforms applicable
to it that land its sector on the canonical partner, and report the set of
corner/master image coefficients available."""
import os, sys, pickle, importlib.util
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import canonicalize_GR as CG

# FIRE masters
spec = importlib.util.spec_from_file_location(
    "grm", os.path.join(ROOT, "topology_input/gravity3L/GR_masters_dict.py"))
grm = importlib.util.module_from_spec(spec); spec.loader.exec_module(grm)
fire = [tuple(m) for s, ms in sorted(grm.GR_MASTERS.items()) for m in ms]

# Kira masters (64)
kira = []
for ln in open(os.path.join(ROOT, "topology_input/gravity3L/kira_validate/results/GR/masters")):
    ln = ln.strip()
    if not ln.startswith("GR["):
        continue
    kira.append(tuple(int(x) for x in ln[3:ln.index("]")].split(",")))
kset = set(kira)
print(f"FIRE masters: {len(fire)}, Kira masters: {len(kira)}")

dropped = [m for m in fire if m not in kset]
print(f"FIRE masters ABSENT from Kira's list: {len(dropped)}")
for m in dropped:
    print(f"  dropped {list(m)}")

cm = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_GR.pkl"), "rb"))
CANON = set(cm["canonical"])


def sec(t):
    return sum(1 << i for i in range(CG.N_DEN) if t[i] > 0)


print("\nper moved FIRE master: available image coefficients (v4 store, direct maps):")
for m in fire:
    S = sec(m)
    if S in CANON:
        continue
    cos = set()
    for (M, c) in CG._transforms(m):
        img = CG.image_unsigned(m, M, c)
        if img is None:
            continue
        # main term: same propagator count
        nb = bin(S).count("1")
        main = [(J, co) for J, co in img.items()
                if bin(sec(J)).count("1") == nb and sec(J) in CANON]
        if len(main) == 1:
            cos.add(main[0][1])
    kdrop = "KIRA-DROPPED" if m in set(dropped) else "kira-kept   "
    print(f"  {kdrop} {list(m)}: image coefficients {sorted(cos)}")
