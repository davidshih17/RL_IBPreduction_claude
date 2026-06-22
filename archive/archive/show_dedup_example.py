#!/usr/bin/env python3
"""Show the 21-state cluster: same expr, different resolved_subs.

Picks 3 representative beam states from the cluster, prints their:
  - expr (same — that's the point)
  - which keys differ between their resolved_subs
  - actual differences in resolved_subs values

This demonstrates concretely why the OLD expr-only dedup was overly
aggressive (would have merged all 21 into 1) and why the NEW resolved_subs
dedup keeps them distinct.
"""
import sys, pickle
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime

ckpath = sys.argv[1] if len(sys.argv) > 1 else \
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/wide_beam_dedup_v1/results/result.pkl.checkpoint'
topo = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

ibp_env.init_from_topology(Topology.from_dir(topo))
set_prime(1009)

with open(ckpath, 'rb') as f:
    ck = pickle.load(f)
beam = ck['beam']

# Group by expr
by_expr = defaultdict(list)
for i, s in enumerate(beam):
    by_expr[frozenset(s.expr.items())].append(i)

# Find the largest cluster
clusters = sorted(by_expr.items(), key=lambda kv: -len(kv[1]))
big_expr_key, big_indices = clusters[0]
print(f'Largest cluster: {len(big_indices)} beam states share THIS SAME expr.')
print(f'Beam indices: {big_indices}')
print()
print('=== The shared expr (all 21 states have this) ===')
sample_state = beam[big_indices[0]]
for k, v in sorted(sample_state.expr.items()):
    if v != 0:
        print(f'  I{list(k)}  coeff={v}')
print()

# Now show that resolved_subs differs across these states
print('=== resolved_subs comparison across cluster members ===')
# Use the first 3 as representatives
i0, i1, i2 = big_indices[0], big_indices[1], big_indices[2]
s0, s1, s2 = beam[i0], beam[i1], beam[i2]

print(f'|resolved_subs| size: state {i0}={len(s0.resolved_subs)}, '
      f'{i1}={len(s1.resolved_subs)}, {i2}={len(s2.resolved_subs)}')

# Find keys present in different subsets of the 3 states
keys0 = set(s0.resolved_subs.keys())
keys1 = set(s1.resolved_subs.keys())
keys2 = set(s2.resolved_subs.keys())

print()
print(f'unique resolved_subs.keys() per state:')
print(f'  state {i0} only: {len(keys0 - keys1 - keys2)}')
print(f'  state {i1} only: {len(keys1 - keys0 - keys2)}')
print(f'  state {i2} only: {len(keys2 - keys0 - keys1)}')
print(f'  shared by all 3: {len(keys0 & keys1 & keys2)}')

# Show a specific key present in state i0 but not i1
only_i0 = keys0 - keys1
if only_i0:
    k = next(iter(only_i0))
    print()
    print(f'=== Example key in state {i0} but NOT state {i1} ===')
    print(f'  resolved_subs[I{list(k)}] in state {i0}:')
    print(f'    |{len(s0.resolved_subs[k])}| terms')
    for kk, vv in list(sorted(s0.resolved_subs[k].items()))[:3]:
        print(f'    I{list(kk)} -> {vv}')
    print(f'  ... (state {i1} does not have this substitution)')

# Also show last actions to see how their paths differ
print()
print('=== last 3 actions of each state ===')
for idx in [i0, i1, i2]:
    s = beam[idx]
    print(f'state {idx}:')
    for t, op, d in s.path[-3:]:
        print(f'  target=I{list(t)}, op={op}, delta={list(d)}')
