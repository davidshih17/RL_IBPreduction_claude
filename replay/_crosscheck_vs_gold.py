"""Adversarial cross-check: standalone replay result vs the sailir-built gold.

Loads:
  - /tmp/replay_result.json            (produced by the standalone replay.py)
  - results/.../replay_state_all4.pkl  ['active_expr']  (the sailir gold)

Asserts the two reduced expressions are IDENTICAL as {integral: coeff} maps
(same key set, same coefficients). Also re-derives PAPER/CORNER/NON counts the
sailir way and checks them against the standalone's classification.

Run:
  python replay/_crosscheck_vs_gold.py > replay/logs/crosscheck.log 2>&1
"""
import json
import os
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_prime, set_paper_masters_only,
                            is_master, weight)

# --- gold (sailir) ---
st = pickle.load(open(os.path.join(
    BASE, 'results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl'), 'rb'))
gold = {tuple(k): v for k, v in st['active_expr'].items() if v != 0}

# --- standalone ---
res = json.load(open('/tmp/replay_result.json'))
mine = {tuple(t['integral']): t['coeff'] for t in res['terms'] if t['coeff'] != 0}

print(f'gold terms       : {len(gold)}')
print(f'standalone terms : {len(mine)}')

only_gold = set(gold) - set(mine)
only_mine = set(mine) - set(gold)
diff_coeff = {k for k in (set(gold) & set(mine)) if gold[k] != mine[k]}

print(f'keys only in gold      : {len(only_gold)}')
print(f'keys only in standalone: {len(only_mine)}')
print(f'shared keys, coeff diff : {len(diff_coeff)}')
for k in list(only_gold)[:5]:
    print(f'  GOLD-ONLY  I{list(k)} = {gold[k]}')
for k in list(only_mine)[:5]:
    print(f'  MINE-ONLY  I{list(k)} = {mine[k]}')
for k in list(diff_coeff)[:5]:
    print(f'  COEFF DIFF I{list(k)} gold={gold[k]} mine={mine[k]}')

expr_identical = (not only_gold) and (not only_mine) and (not diff_coeff)

# --- classification cross-check (sailir is_master vs standalone categories) ---
init_from_topology(Topology.from_dir(os.path.join(BASE, 'topology_input/pentagonbox')))
set_prime(1009)
set_paper_masters_only(False)
MS = ibp_env.MASTERS_SET

g_paper = sum(1 for k in gold if k in MS)
g_corner = sum(1 for k in gold if k not in MS and is_master(k))
g_non = sum(1 for k in gold if not is_master(k))
print(f'\nsailir classification of gold: {g_paper} PAPER + {g_corner} CORNER + {g_non} NON')
print(f'standalone classification    : {res["n_paper"]} PAPER + '
      f'{res["n_corner"]} CORNER + {res["n_non"]} NON')

class_match = (g_paper == res['n_paper'] and g_corner == res['n_corner']
               and g_non == res['n_non'])

print('\n' + ('CROSSCHECK PASS: expressions identical AND classifications match'
              if (expr_identical and class_match)
              else 'CROSSCHECK FAIL'))
sys.exit(0 if (expr_identical and class_match) else 1)
