"""
Pre-flight check for pentagon-box (TA) training data setup.

For the 8+3 propagator/ISP split (D1..D8 denominators, D9..D11 ISPs):
  - enumerate all 2^8 - 1 = 255 candidate "corner" integrals over D1..D8;
  - check which of those sectors are non-trivial per Kira (i.e., produce
    valid integrals at all). Trivial sectors have no integrals and SAILIR
    should not generate training trajectories from them;
  - check whether the corner integral itself is one of Kira's 61 masters.

Inputs:
  - Kira's nonTrivialSector  (sector list with t)
  - Kira's results/TA/masters (the 61-master basis)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

KIRA_DIR = Path("/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples/2-loop-pentagonbox")

# ---- parse Kira's non-trivial sectors (all sectors over 11 indices) ----
nontrivial_secs = {}  # sector_id -> t
with open(KIRA_DIR / "sectormappings/TA/nonTrivialSector") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        sec, t = ln.split()
        nontrivial_secs[int(sec)] = int(t)
print(f"Kira non-trivial sectors (over 11 indices): {len(nontrivial_secs)}")

# Filter to those that are pure 8-bit sectors (bits 8,9,10 unset)
# These correspond to "physical" TA sectors with D9..D11 only as ISPs.
phys_secs = {s: t for s, t in nontrivial_secs.items() if s < 256}
print(f"  ... with no bit 8/9/10 set (sector < 256): {len(phys_secs)}")

# ---- parse Kira's 61 masters ----
kira_masters = []
with open(KIRA_DIR / "results/TA/masters") as f:
    for ln in f:
        m = re.match(r'TA\[([^\]]+)\]', ln.strip())
        if m:
            kira_masters.append(tuple(int(x.strip()) for x in m.group(1).split(',')))
print(f"Kira masters: {len(kira_masters)}")

def sector_of(integral):
    return sum((1 << i) for i, a in enumerate(integral) if a > 0)

# Sectors that have masters
sectors_with_masters = sorted({sector_of(m) for m in kira_masters})
masters_by_sec = {}
for m in kira_masters:
    masters_by_sec.setdefault(sector_of(m), []).append(m)
print(f"Sectors that contain masters: {len(sectors_with_masters)}")
print(f"  all bits 0-7 only (in phys hierarchy):")
phys_master_secs = [s for s in sectors_with_masters if s < 256]
print(f"  {phys_master_secs}")

# ---- enumerate all 255 candidate corner integrals over D1..D8 ----
def corner_of(sector_id, n=11):
    return tuple(1 if (sector_id >> i) & 1 else 0 for i in range(n))

candidates = []
for s in range(1, 256):
    c = corner_of(s)
    t = sum(1 for x in c if x > 0)
    candidates.append((s, t, c))

# Cross-check: how many of the 255 candidates fall on a Kira-non-trivial sector?
phys_set = set(phys_secs)
viable = [(s,t,c) for (s,t,c) in candidates if s in phys_set]
trivial_per_kira = [(s,t,c) for (s,t,c) in candidates if s not in phys_set]
print()
print(f"Of the 255 candidate 8-bit corner sectors:")
print(f"  viable (non-trivial per Kira): {len(viable)}")
print(f"  trivial (Kira says no integrals here): {len(trivial_per_kira)}")

# Are any corners ALREADY in the master list?
master_set = set(kira_masters)
corner_is_master = [(s,t,c) for (s,t,c) in viable if c in master_set]
print(f"  corner IS itself a Kira master: {len(corner_is_master)}")
for s, t, c in corner_is_master:
    print(f"    sec {s:>3} (t={t}): {c}")

# Distribution by t
print()
print("Distribution of viable corners by t:")
viable_t = Counter(t for s,t,c in viable)
trivial_t = Counter(t for s,t,c in trivial_per_kira)
print(f"  {'t':>3}  {'viable':>6}  {'trivial':>7}")
for t in sorted(set(viable_t) | set(trivial_t)):
    print(f"  {t:>3}  {viable_t.get(t,0):>6}  {trivial_t.get(t,0):>7}")

# Examples
print()
print(f"Examples of trivial corners (first 10):")
for s, t, c in trivial_per_kira[:10]:
    print(f"  sec {s:>3} (t={t}): {c}")
print()
print(f"Examples of viable corners (first 5):")
for s, t, c in viable[:5]:
    in_basis = ' [IS A MASTER]' if c in master_set else ''
    has_master_in_sec = ' [sec has masters]' if s in masters_by_sec else ''
    print(f"  sec {s:>3} (t={t}): {c}{in_basis}{has_master_in_sec}")
