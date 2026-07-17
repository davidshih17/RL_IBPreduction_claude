#!/usr/bin/env python
"""Select a ~30-integral GR benchmark from the ~2100 FIRE-table targets, with a
deliberate spread in difficulty:
  - denominator count L (6..10) — the histogram is 3 / 202 / 873 / 898 / 125
  - weight (r, s): r = sum of positive indices, s = sum |negative| (ISP degree +
    numerator powers on absent props)
  - sector: canonical vs non-canonical (exercises the router), including sectors
    whose FIRE masters get relabeled (the canonical-masters path)
Output: results/gr_benchmark_targets.txt (one CSV integral per line) + a summary
table on stdout for approval. SELECTION ONLY — nothing is launched.
"""
import os, re, pickle, random
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
SRC = os.path.join(BASE, "topology_input/fire_tables40_math_All.m")
random.seed(20260717)

text = open(SRC).read()
t_start = text.index("tableAll[40]")
targets = sorted({tuple(int(x) for x in g.split(","))
                  for g in re.findall(r"FF\[40,\s*\{([^}]*)\}\]", text[t_start:])})
print(f"distinct FIRE targets: {len(targets)}")

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_GR.pkl"), "rb"))
CANON = set(cm["canonical"]); REP = cm["rep_of"]
RELABEL_SECTORS = {762, 765, 895, 1018, 1021}      # FIRE masters relabeled from here


def feats(t):
    r = sum(x for x in t if x > 0)
    s = sum(-x for x in t if x < 0)
    sec = sum(1 << i for i in range(10) if t[i] > 0)
    L = bin(sec).count("1")
    return r, s, sec, L


rows = [(t, *feats(t)) for t in targets]

# difficulty bands: (L, r+s) — stratify
from collections import defaultdict
bands = defaultdict(list)
for t, r, s, sec, L in rows:
    band = (L, min((r + s - L) // 2, 4))        # excess weight above corner, coarse
    bands[band].append((t, r, s, sec, L))

picked = []
# 1) all 6-den targets (easiest, tiny)
picked += [x for x in rows if x[4] == 6]
# 2) per L in 7..10: lightest, median, heaviest by (r+s), plus one non-canonical
for L in (7, 8, 9, 10):
    grp = sorted([x for x in rows if x[4] == L], key=lambda x: (x[1] + x[2], x[0]))
    if not grp:
        continue
    sel = [grp[0], grp[len(grp) // 2], grp[-1]]
    noncanon = [x for x in grp if x[3] not in CANON]
    if noncanon:
        sel.append(noncanon[len(noncanon) // 2])
    relab = [x for x in grp if x[3] in RELABEL_SECTORS]
    if relab:
        sel.append(relab[0])
    # one extra mid-heavy for spread
    sel.append(grp[3 * len(grp) // 4])
    picked += sel
# 3) a couple of high-s (deep ISP) cases
deep = sorted(rows, key=lambda x: -x[2])[:3]
picked += deep

# dedup, sort by difficulty
seen = set(); bench = []
for x in picked:
    if x[0] not in seen:
        seen.add(x[0]); bench.append(x)
bench.sort(key=lambda x: (x[4], x[1] + x[2]))

print(f"\nselected: {len(bench)} integrals")
print(f"{'integral':<52} {'L':>2} {'r':>3} {'s':>2} {'sector':>6} {'canon':>6} {'relabel':>7}")
for t, r, s, sec, L in bench:
    print(f"{str(list(t)):<52} {L:>2} {r:>3} {s:>2} {sec:>6} "
          f"{'yes' if sec in CANON else 'NO':>6} "
          f"{'yes' if sec in RELABEL_SECTORS else '':>7}")

with open(os.path.join(BASE, "results/gr_benchmark_targets.txt"), "w") as f:
    for t, *_ in bench:
        f.write(",".join(str(x) for x in t) + "\n")
print(f"\nsaved -> results/gr_benchmark_targets.txt")
from collections import Counter
print("spread:", dict(Counter(x[4] for x in bench)), "| non-canonical:",
      sum(1 for x in bench if x[3] not in CANON), "| max (r,s):",
      max((x[1], x[2]) for x in bench))
