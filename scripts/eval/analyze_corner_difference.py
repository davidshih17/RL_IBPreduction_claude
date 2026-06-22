"""Analyze the difference between two completed (8,5) reductions.

D = Fresh - Gold  (coefficient-wise, mod prime), where
  Fresh = results/pentagonbox_8_5_v7_fresh/replay_state.pkl   ['active_expr']  (253 terms)
  Gold  = results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl ['active_expr'] (262 terms)

Both equal the SAME start integral I[1,1,1,1,1,1,1,1,-5,0,0], so D must vanish as an
integral identity. This script checks:
  (1) D restricted to the 61 PAPER masters is exactly zero  -> the true basis agrees.
  (2) D is supported ONLY on corner integrals.
  (3) prints D's corner terms grouped by level (popcount of the propagator sector) and
      annotates each with its present-propagator set, so symmetry / scaleless structure
      is visible.

Propagator legend (TA pentagonbox, integralfamilies.yaml):
  0:k1  1:k1+p1  2:k1+p1+p2  3:k1+p1+p2+p3  4:k2  5:k2+p1+p2+p3  6:k2+p1+p2+p3+p4  7:k1-k2
"""
import os
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
import pickle
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_prime, set_paper_masters_only,
                            is_master, weight, get_sector)

PRIME = 1009
START = (1, 1, 1, 1, 1, 1, 1, 1, -5, 0, 0)
FRESH = os.path.join(BASE, 'results/pentagonbox_8_5_v7_fresh/replay_state.pkl')
GOLD = os.path.join(BASE, 'results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl')

init_from_topology(Topology.from_dir(os.path.join(BASE, 'topology_input/pentagonbox')))
set_prime(PRIME)
set_paper_masters_only(False)
MS = ibp_env.MASTERS_SET

fresh = {tuple(k): v % PRIME for k, v in pickle.load(open(FRESH, 'rb'))['active_expr'].items() if v % PRIME}
gold = {tuple(k): v % PRIME for k, v in pickle.load(open(GOLD, 'rb'))['active_expr'].items() if v % PRIME}
print(f'Fresh: {len(fresh)} terms   Gold: {len(gold)} terms')

# D = Fresh - Gold (mod PRIME)
keys = set(fresh) | set(gold)
D = {}
for k in keys:
    d = (fresh.get(k, 0) - gold.get(k, 0)) % PRIME
    if d:
        D[k] = d
print(f'\nD = Fresh - Gold : {len(D)} nonzero terms')

# (1) paper-master part of D
paper_D = {k: c for k, c in D.items() if k in MS}
print(f'  of which PAPER (true masters): {len(paper_D)}  '
      f'-> {"ALL paper coeffs identical (D_paper = 0)" if not paper_D else "MISMATCH!"}')
for k, c in sorted(paper_D.items()):
    print(f'    PAPER DIFF I{list(k)} = {c}')

# (2) is the rest all corners?
non_master_D = {k: c for k, c in D.items() if not is_master(k)}
print(f'  of which NON-master: {len(non_master_D)}  '
      f'-> {"none (good)" if not non_master_D else "PRESENT!"}')
corner_D = {k: c for k, c in D.items() if k not in MS and is_master(k)}
print(f'  of which CORNER integrals: {len(corner_D)}')

# (3) group corner part of D by level (popcount of propagator sector)
def present_props(k):
    return [p for p in range(8) if k[p] > 0]   # first 8 are denominators

by_level = {}
for k, c in corner_D.items():
    lvl = sum(get_sector(k))
    by_level.setdefault(lvl, []).append((k, c))

print('\n' + '=' * 78)
print('CORNER terms of D = Fresh - Gold, grouped by level (#propagators)')
print('=' * 78)
for lvl in sorted(by_level, reverse=True):
    rows = sorted(by_level[lvl], key=lambda kc: kc[0])
    print(f'\n--- level {lvl}  ({len(rows)} corner terms) ---')
    for k, c in rows:
        src = 'fresh-only' if k not in gold else ('gold-only' if k not in fresh else 'both')
        print(f'  D={c:>4}  props={present_props(k)}  sector_id={ibp_env.get_sector_id(k)}  '
              f'[{src}]  I{list(k)}')

# Also: net count change at each level (fresh - gold corner counts)
print('\n' + '=' * 78)
print('Corner-count by level: Gold vs Fresh')
print('=' * 78)
def corner_levels(expr):
    d = {}
    for k in expr:
        if k not in MS and is_master(k):
            d[sum(get_sector(k))] = d.get(sum(get_sector(k)), 0) + 1
    return d
gl, fl = corner_levels(gold), corner_levels(fresh)
for lvl in sorted(set(gl) | set(fl), reverse=True):
    print(f'  level {lvl}: gold={gl.get(lvl,0):>3}  fresh={fl.get(lvl,0):>3}  '
          f'delta={fl.get(lvl,0)-gl.get(lvl,0):+d}')
print('\nANALYZE DONE')
