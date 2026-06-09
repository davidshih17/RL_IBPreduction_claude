#!/usr/bin/env python3
"""Inspect the current beam states from a worker's live checkpoint.

Loads the .pkl.checkpoint file, prints per-state summary:
    state_idx, max_weight, n_non_masters, n_terms_in_expr, score, last action

Usage:
    python inspect_beam_checkpoint.py <path/to/checkpoint>
"""
import sys
import pickle
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime, weight, is_master
from beam_search_utils import get_sector_mask

if len(sys.argv) < 2:
    sys.exit("Usage: inspect_beam_checkpoint.py <path/to/checkpoint> [topology_dir] [integral]")
ckpath = sys.argv[1]
topo = sys.argv[2] if len(sys.argv) > 2 else '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'
integral_str = sys.argv[3] if len(sys.argv) > 3 else '-1,2,1,0,1,2,1,1,-3,0,0'

ibp_env.init_from_topology(Topology.from_dir(topo))
set_prime(1009)
target_sector = tuple(get_sector_mask(tuple(int(x) for x in integral_str.split(','))))

with open(ckpath, 'rb') as f:
    ck = pickle.load(f)

print(f'Checkpoint: {ckpath}')
print(f'Step:               {ck["step"]}')
print(f'best_weight_ever:   {ck["best_weight_ever"]}')
print(f'initial_weight:     {ck["initial_weight"]}')
print(f'beam size:          {len(ck["beam"])}')
print(f'steps_since_improvement: {ck.get("steps_since_improvement", 0)}')
print()

def max_w(expr):
    nms = [k for k, v in expr.items() if v != 0 and not is_master(k)
           and tuple(get_sector_mask(k)) == target_sector]
    if not nms:
        return (0, 0)
    return max((weight(k)[0], weight(k)[1]) for k in nms)

def n_nonmasters_in_sector(expr):
    return sum(1 for k, v in expr.items() if v != 0 and not is_master(k)
               and tuple(get_sector_mask(k)) == target_sector)

print(f"{'idx':>4s} {'max_w':>9s} {'n_nms':>6s} {'expr':>5s} {'subs':>5s} {'score':>9s} | last action(s)")
print('-' * 90)
states = list(enumerate(ck['beam']))
# Sort by max_weight ascending so the "best" is first
states.sort(key=lambda kv: (max_w(kv[1].expr), n_nonmasters_in_sector(kv[1].expr)))
for orig_idx, s in states:
    mw = max_w(s.expr)
    nm = n_nonmasters_in_sector(s.expr)
    last_actions = s.path[-3:] if len(s.path) >= 3 else s.path
    actions_str = ' '.join(f'(t=I{list(t)[:6]}...,op={op})' for t, op, d in last_actions[-2:])
    print(f"{orig_idx:4d} {str(mw):>9s} {nm:6d} {len(s.expr):5d} {len(s.subs):5d} {s.score:9.3f} | ...{actions_str[-80:]}")

print()
# Distribution of max_weights across beam
from collections import Counter
mw_counts = Counter(max_w(s.expr) for s in ck['beam'])
print('max_weight distribution across beam:')
for mw, n in sorted(mw_counts.items()):
    print(f'  {str(mw):>10s}: {n}')
print()
nm_counts = [n_nonmasters_in_sector(s.expr) for s in ck['beam']]
print(f'n_non_masters in target_sector across beam: min={min(nm_counts)}, max={max(nm_counts)}, '
      f'median={sorted(nm_counts)[len(nm_counts)//2]}, mean={sum(nm_counts)/len(nm_counts):.1f}')

# How many unique exprs (deduped by frozenset of expr keys)?
expr_hashes = set()
for s in ck['beam']:
    expr_hashes.add(frozenset(s.expr.items()))
print(f'unique exprs (by content): {len(expr_hashes)} / {len(ck["beam"])}')

# How many unique (sub-sector projections of expr)?
sect_hashes = set()
for s in ck['beam']:
    proj = frozenset(
        (k, v) for k, v in s.expr.items()
        if v != 0 and tuple(get_sector_mask(k)) == target_sector
    )
    sect_hashes.add(proj)
print(f'unique target-sector projections: {len(sect_hashes)} / {len(ck["beam"])}')
