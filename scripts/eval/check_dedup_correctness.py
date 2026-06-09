#!/usr/bin/env python3
"""Verify the resolved_subs-based dedup:
  - All beam states should have UNIQUE resolved_subs.
  - Show how many have same expr but different resolved_subs (these should
    be kept as distinct under the new key — they were merged by the old
    expr-only key).
"""
import sys, pickle, hashlib
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime

ckpath = sys.argv[1]
topo = sys.argv[2] if len(sys.argv) > 2 else '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

ibp_env.init_from_topology(Topology.from_dir(topo))
set_prime(1009)

with open(ckpath, 'rb') as f:
    ck = pickle.load(f)
beam = ck['beam']
N = len(beam)
print(f'checkpoint step={ck["step"]}, beam_size={N}\n')

def resolved_key(s):
    return frozenset((k, frozenset(v.items())) for k, v in s.resolved_subs.items())

def expr_key(s):
    return frozenset(s.expr.items())

resolved_keys = [resolved_key(s) for s in beam]
expr_keys = [expr_key(s) for s in beam]

print(f'Unique resolved_subs fingerprints: {len(set(resolved_keys))} / {N}')
print(f'Unique expr fingerprints:          {len(set(expr_keys))} / {N}')
print()

# Group by expr; show how many distinct resolved_subs share the same expr.
by_expr = defaultdict(list)
for i, ek in enumerate(expr_keys):
    by_expr[ek].append(i)

shared = [(k, idxs) for k, idxs in by_expr.items() if len(idxs) > 1]
print(f'Expr-keys with >1 beam state attached: {len(shared)}')
for k, idxs in shared[:5]:
    # Check if their resolved_subs differ
    distinct = len({resolved_keys[i] for i in idxs})
    print(f'  Expr (size {len(k)}) shared by {len(idxs)} states; '
          f'distinct resolved_subs = {distinct}')

# Pick an example pair: same expr, different resolved_subs
example = None
for k, idxs in shared:
    if len({resolved_keys[i] for i in idxs}) > 1:
        example = idxs
        break

if example is None:
    print('\nNo states share the same expr — dedup is doing its job, all 20 beam')
    print('states are distinct at the expr level too. (Stronger condition than required.)')
else:
    i, j = example[0], example[1]
    si, sj = beam[i], beam[j]
    print(f'\nExample of same expr but different resolved_subs: states {i} and {j}')
    print(f'  expr keys (sample): {sorted(si.expr.keys())[:3]}')
    print(f'  state {i}: |subs|={len(si.subs)}, last action={si.path[-1] if si.path else None}')
    print(f'  state {j}: |subs|={len(sj.subs)}, last action={sj.path[-1] if sj.path else None}')

    # Find a key that's in both resolved_subs with different values
    common_keys = set(si.resolved_subs.keys()) & set(sj.resolved_subs.keys())
    for k in sorted(common_keys):
        if si.resolved_subs[k] != sj.resolved_subs[k]:
            print(f'\n  Difference at resolved_subs[I{list(k)}]:')
            print(f'    state {i} value: |{len(si.resolved_subs[k])}| terms, sample: {dict(list(si.resolved_subs[k].items())[:3])}')
            print(f'    state {j} value: |{len(sj.resolved_subs[k])}| terms, sample: {dict(list(sj.resolved_subs[k].items())[:3])}')
            break

    # Also count subs differences
    only_i = set(si.resolved_subs.keys()) - set(sj.resolved_subs.keys())
    only_j = set(sj.resolved_subs.keys()) - set(si.resolved_subs.keys())
    print(f'\n  resolved_subs.keys() comparison:')
    print(f'    in {i} only: {len(only_i)}')
    print(f'    in {j} only: {len(only_j)}')
    print(f'    in both:     {len(common_keys)}')
