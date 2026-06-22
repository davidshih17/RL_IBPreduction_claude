"""Round 2 setup: load the round-1 replay expression (active_expr) and identify
the NON-MASTERS that remain to be reduced.

Masters (per the user's definition, == is_master() with PAPER_MASTERS_ONLY=False):
  paper/Kira masters (topology.masters == MASTERS_SET)  UNION
  corner integrals of sectors NOT covered by the master basis.

Non-masters = active_expr terms that are not masters. These are exactly the
integrals round-1 left un-reduced (failed/trapped → cached as identity), read
straight off the expression rather than a separate failure list.
"""
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_paper_masters_only,
                            set_prime, is_master, is_corner_integral, get_sector)

TOPO = f'{BASE}/topology_input/pentagonbox'
STATE = f'{BASE}/results/pentagonbox_8_5_v6/replay_state.pkl'
OUT = f'{BASE}/results/pentagonbox_8_5_v6/round2_nonmasters.pkl'

topology = Topology.from_dir(TOPO)
init_from_topology(topology)
set_prime(1009)
set_paper_masters_only(False)   # masters = paper masters UNION corner integrals

MS = ibp_env.MASTERS_SET
print(f'topology masters (MASTERS_SET): {len(MS)}')

state = pickle.load(open(STATE, 'rb'))
active_expr = state['active_expr']
start = state['start_integral']
print(f'start_integral : {start}')
print(f'built_at       : {state.get("built_at")}')
print(f'round1 success={state.get("n_success")} fail={state.get("n_fail")} '
      f'pkls={state.get("num_pkls_scanned")}')
print(f'active_expr terms (distinct integrals): {len(active_expr)}')
print()

masters_paper = []
masters_corner = []
nonmasters = []
for ig, coeff in active_expr.items():
    if ig in MS:
        masters_paper.append((ig, coeff))
    elif is_master(ig):           # corner integral in an uncovered sector
        masters_corner.append((ig, coeff))
    else:
        nonmasters.append((ig, coeff))

print('=== classification of active_expr terms ===')
print(f'  paper/Kira masters (in MASTERS_SET) : {len(masters_paper)}')
print(f'  corner-integral masters (uncovered) : {len(masters_corner)}')
print(f'  NON-MASTERS (need round 2)          : {len(nonmasters)}')
print(f'  total check: '
      f'{len(masters_paper)+len(masters_corner)+len(nonmasters)} == '
      f'{len(active_expr)}')
print()

# Sanity: how many non-masters are themselves corner integrals (i.e. corners in
# COVERED sectors, which reduce to the paper master and so are not masters)?
nm_corner = [(ig, c) for ig, c in nonmasters if is_corner_integral(ig)]
print(f'  of which are corner integrals (covered sectors): {len(nm_corner)}')

# Distribution of non-masters by sector weight (w1,w2) for a sense of scale.
from collections import Counter
wc = Counter()
for ig, c in nonmasters:
    sec = get_sector(ig)          # 0/1 propagator mask (tuple)
    wc[sum(sec)] += 1
print(f'  non-masters by #propagators in sector: '
      f'{dict(sorted(wc.items()))}')
print()

# Save the non-master list for the round-2 orchestrator.
nm_list = [ig for ig, c in nonmasters]
pickle.dump({
    'nonmasters': nm_list,
    'nonmasters_with_coeff': nonmasters,
    'start_integral': start,
    'source_state': STATE,
    'n_paper_masters': len(masters_paper),
    'n_corner_masters': len(masters_corner),
}, open(OUT, 'wb'))
print(f'wrote {OUT}  ({len(nm_list)} non-masters)')
print()

print('=== sample non-masters (first 25, sorted) ===')
for ig, c in sorted(nonmasters)[:25]:
    print(f'  {ig}  coeff={c}')
