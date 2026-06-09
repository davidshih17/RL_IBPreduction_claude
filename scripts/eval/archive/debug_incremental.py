#!/usr/bin/env python3
"""Minimal repro: focus on a single (sub_int, op, shift) entry and compare
cached dict between full rebuild and incremental.
"""
import os
import sys
import pickle
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    compute_indirect_substituted_with_aux,
    apply_resolved_subs, apply_resolved_subs_batch,
    add_sub_to_resolved,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
    get_raw_equation,
)
from beam_search_utils import get_sector_mask

t = Topology.from_dir('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox')
ibp_env.init_from_topology(t)
set_prime(1009)
env = IBPEnvironment()

with open('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v3/work/results/straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint', 'rb') as f:
    ck = pickle.load(f)

s0 = ck['beam'][0]
INTEGRAL = (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)
target_sector = tuple(get_sector_mask(INTEGRAL))
fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)

# Determinism check: call compute twice on the same state.
print("=== Determinism check ===")
import time
t0 = time.time()
res_a, _ = compute_indirect_substituted_with_aux(
    fsubs, fresolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'call A done in {time.time()-t0:.2f}s')
t0 = time.time()
res_b, _ = compute_indirect_substituted_with_aux(
    fsubs, fresolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'call B done in {time.time()-t0:.2f}s')

mm = 0
for (a, b) in zip(res_a, res_b):
    if a[4] != b[4] or a[5] != b[5]:
        mm += 1
print(f'mismatches A vs B: {mm} (should be 0 for deterministic compute)')

# Now test apply_resolved_subs_batch directly
print('\n=== apply_resolved_subs_batch determinism ===')
fake_target = (1, 2, 1, 0, 2, 2, 1, 1, -2, -1, 0)
fake_sol = dict(list(fresolved.values())[0])
new_resolved = add_sub_to_resolved(fresolved, fake_target, fake_sol)

# Build one raw via the same formula
def grc(op, seed):
    key = (op, seed)
    if key not in env._raw_eq_cache:
        env._raw_eq_cache[key] = get_raw_equation(env.ibp_t, env.li_t, op, seed)
    return env._raw_eq_cache[key]

# pick a shift from env.shifts[0]
op0 = list(env.shifts.keys())[0]
shift0 = env.shifts[op0][0]
seed = tuple(fake_target[i] - shift0[i] for i in range(11))
raw = grc(op0, seed)
print(f'fake_target={fake_target}, op={op0}, shift={shift0}, seed={seed}')
print(f'|raw|={len(raw)}, fake_target in raw: {fake_target in raw}')

# Apply new_resolved to raw twice
c1 = apply_resolved_subs_batch([raw], new_resolved)[0]
c2 = apply_resolved_subs_batch([raw], new_resolved)[0]
print(f'apply twice — c1 == c2: {c1 == c2}, |c1|={len(c1)}, |c2|={len(c2)}')

# Now construct new_subs with fake AT END (after adding to dict), then run full rebuild
new_subs = dict(fsubs); new_subs[fake_target] = fake_sol
print(f'\n=== Full rebuild with fake sub appended ===')
res_full, _ = compute_indirect_substituted_with_aux(
    new_subs, new_resolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)

# Find the entry matching (fake_target, op0, shift0) in res_full
match = [e for e in res_full if e[0] == fake_target and e[1] == op0 and e[2] == shift0]
print(f'found {len(match)} matching entries in full rebuild')
if match:
    e = match[0]
    print(f'full rebuild cached for this entry: {sorted(e[4].items())[:3]} ...')
    print(f'directly-computed cached:           {sorted(c1.items())[:3]} ...')
    print(f'directly == full?                   {e[4] == c1}')
