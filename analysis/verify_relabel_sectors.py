#!/usr/bin/env python
"""Verify which sectors house FIRE masters in NON-canonical sectors (i.e. the
sectors whose masters get relabeled by canonicalization). Compares against the
hardcoded set used in select_gr_benchmark.py."""
import os, re, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"

text = open(os.path.join(BASE, "topology_input/fire_tables40_math_All.m")).read()
m = re.search(r"masterAll\[40\]\s*=\s*\{(.*?)\}\s*(?:\n\s*\n|$)", text, re.S)
masters = [tuple(int(x) for x in g.split(","))
           for g in re.findall(r"G\[40,\s*\{([^}]*)\}\]", m.group(1))]
print(f"FIRE masters: {len(masters)}")

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_GR.pkl"), "rb"))
CANON = set(cm["canonical"])

noncanon_secs = {}
for t in masters:
    sec = sum(1 << i for i in range(10) if t[i] > 0)
    if sec not in CANON:
        noncanon_secs.setdefault(sec, []).append(t)

print(f"masters in NON-canonical sectors: "
      f"{sum(len(v) for v in noncanon_secs.values())}")
for sec in sorted(noncanon_secs):
    print(f"  sector {sec}: {len(noncanon_secs[sec])} masters -> canonical rep "
          f"{cm['rep_of'][sec]}")

hardcoded = {762, 765, 895, 1018, 1021}
actual = set(noncanon_secs)
print(f"\nhardcoded set in select_gr_benchmark.py: {sorted(hardcoded)}")
print(f"actual set:                              {sorted(actual)}")
print(f"MATCH: {hardcoded == actual}")
if hardcoded != actual:
    print(f"  missing from hardcoded: {sorted(actual - hardcoded)}")
    print(f"  wrongly in hardcoded:   {sorted(hardcoded - actual)}")
