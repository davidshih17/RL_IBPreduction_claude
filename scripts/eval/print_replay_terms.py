"""Print every term of a saved replay_state's active_expr (the result of replaying
the start integral through the combined cache), classified as
PAPER-master / CORNER-master / NON-master, sorted heaviest-first.

Usage: print_replay_terms.py <replay_state.pkl>
"""
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_paper_masters_only,
                            set_prime, is_master, weight)

path = sys.argv[1]
init_from_topology(Topology.from_dir(f'{BASE}/topology_input/pentagonbox'))
set_prime(1009)
set_paper_masters_only(False)            # masters = paper UNION corner integrals

st = pickle.load(open(path, 'rb'))
start = st['start_integral']
expr = st['active_expr']
MS = ibp_env.MASTERS_SET


def cat(ig):
    if ig in MS:
        return 'PAPER '
    if is_master(ig):
        return 'CORNER'
    return 'NONMAS'


terms = [(ig, c) for ig, c in expr.items() if c != 0]
# Heaviest first: by (r, s, abs-tuple), then category.
terms.sort(key=lambda kv: (-weight(kv[0])[0], -weight(kv[0])[1], kv[0]))

print(f'REPLAY of START I{list(start)}  ->  {len(terms)} terms')
print(f'{"#":>4} {"coeff":>5} {"cat":>6} {"w":>8}  integral')
print('-' * 70)
n_paper = n_corner = n_nm = 0
for i, (ig, c) in enumerate(terms):
    k = cat(ig)
    if k == 'PAPER ':
        n_paper += 1
    elif k == 'CORNER':
        n_corner += 1
    else:
        n_nm += 1
    w = weight(ig)
    print(f'{i+1:>4} {c:>5} {k} ({w[0]},{w[1]})  I{list(ig)}')
print('-' * 70)
print(f'TOTAL {len(terms)} terms = {n_paper} PAPER + {n_corner} CORNER masters '
      f'+ {n_nm} NON-masters')
