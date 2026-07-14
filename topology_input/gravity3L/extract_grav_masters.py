#!/usr/bin/env python3
"""Extract the gravity-3L master basis (FIRE family 40) from
topology_input/fire_tables40_math_All.m into SAILIR format.

FIRE family: 3 loops (k1,k2,k3), 15 indices. Denominators = positions 1-10
(1-indexed); ISPs = positions 11-15 (confirmed always <=0 across all masters).
masterAll[40] holds 68 masters; tableAll[40] holds the full reduction.

Writes (into this directory):
  masters              — SAILIR master list  "GR[i1,...,i15]  # sector_id"
  GR_masters_dict.py   — {sector_id: [master tuples]} + topology constants

Also scans the reduction-table LHS to report the top sector (max denominators)
for the integralfamilies.yaml that step 2 will need.
"""
import re
from pathlib import Path
from collections import defaultdict

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
FIRE = BASE / 'topology_input/fire_tables40_math_All.m'
HERE = Path(__file__).resolve().parent
FAMILY = 'GR'
N_INDICES = 15
N_DENOMINATORS = 10
ISP_POSITIONS = (10, 11, 12, 13, 14)   # 0-indexed -> 1-indexed positions 11..15

text = FIRE.read_text()

# --- masters: parse the masterAll[40] = { G[40,{..}], ... } block ---
mstart = text.index('masterAll[40]')
mend = text.index('tableAll[40]')
mblock = text[mstart:mend]
masters = []
for mm in re.finditer(r'G\[40,\s*\{\s*([0-9,\s\-]+?)\s*\}\s*\]', mblock):
    v = tuple(int(x) for x in mm.group(1).replace(' ', '').split(','))
    if len(v) == N_INDICES:
        masters.append(v)
print(f"parsed {len(masters)} masters", flush=True)

# sanity: ISP slots must be <=0 in every master
for v in masters:
    assert all(v[i] <= 0 for i in ISP_POSITIONS), f"master violates ISP convention: {v}"


def sector_of(v):
    return sum((1 << i) for i, a in enumerate(v) if a > 0)


# --- write masters file ---
with open(HERE / 'masters', 'w') as f:
    for v in masters:
        f.write(f"{FAMILY}[{','.join(map(str, v))}]  # {sector_of(v)}\n")

# --- write GR_masters_dict.py ---
by_sec = defaultdict(list)
for v in masters:
    by_sec[sector_of(v)].append(v)
with open(HERE / 'GR_masters_dict.py', 'w') as f:
    f.write('"""\nGravity 3-loop (FIRE family 40 -> SAILIR family GR) master basis,\n')
    f.write('extracted from topology_input/fire_tables40_math_All.m (masterAll[40]).\n\n')
    f.write('Same schema as trianglebox PAPER_MASTERS: sector_id -> list of length-15 tuples.\n"""\n\n')
    f.write('GR_MASTERS = {\n')
    for sec in sorted(by_sec):
        f.write(f"    {sec}: [\n")
        for v in by_sec[sec]:
            f.write(f"        {v},\n")
        f.write("    ],\n")
    f.write('}\n\n')
    f.write('# 10 denominators (positions 1-10) + 5 ISPs (positions 11-15).\n')
    f.write(f'ISP_POSITIONS = {ISP_POSITIONS}\n')
    f.write(f'N_INDICES = {N_INDICES}\n')
    f.write(f'N_DENOMINATORS = {N_DENOMINATORS}\n')

print(f"wrote masters ({len(masters)}) and GR_masters_dict.py; sectors with masters: {len(by_sec)}", flush=True)
print(f"master denominator-count range: {min(bin(sector_of(v)).count('1') for v in masters)}.."
      f"{max(bin(sector_of(v)).count('1') for v in masters)}", flush=True)

# --- scan reduction-table integrals to find the TOP sector (max denominators) ---
# table entries look like  FF[40, {15 ints}] -> ...   (both LHS targets and RHS)
best = 0
best_sec = 0
best_int = None
seen_pop = defaultdict(int)
for mm in re.finditer(r'FF\[40,\s*\{\s*([0-9,\s\-]+?)\s*\}\s*\]', text):
    v = tuple(int(x) for x in mm.group(1).replace(' ', '').split(','))
    if len(v) != N_INDICES:
        continue
    # denominators present = positions 0..9 with index > 0
    dens = sum(1 for i in range(N_DENOMINATORS) if v[i] > 0)
    seen_pop[dens] += 1
    if dens > best:
        best, best_sec, best_int = dens, sector_of(v), v
print(f"\nTABLE SCAN: top integral has {best} denominators present", flush=True)
print(f"  top sector id = {best_sec}  (= bits {bin(best_sec)})", flush=True)
print(f"  example top integral: {FAMILY}{list(best_int)}", flush=True)
print(f"  denominator-count histogram (count of distinct table integrals by #denoms):", flush=True)
for k in sorted(seen_pop):
    print(f"    {k} denoms: {seen_pop[k]}", flush=True)
