#!/usr/bin/env python3
"""Correctness test for compute_indirect_substituted_incremental.

Strategy:
1. Load the (8,4) production checkpoint at step N.
2. Compute indirect_cache from scratch (full rebuild) at step N — call this baseline_full.
3. Apply ONE action to the first beam state: get new_subs, new_resolved_subs.
4. Method A: full rebuild on new state → expected.
5. Method B: incremental from baseline_full's aux_state, with the new sub → got.
6. Compare expected vs got, key by key, value by value.

If they match bit-exact, the incremental optimization is correct.
"""
import os
import sys
import pickle
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, get_raw_equation,
    compute_indirect_substituted, compute_indirect_substituted_with_aux,
    compute_indirect_substituted_incremental,
    apply_resolved_subs, add_sub_to_resolved, solve_ibp_for,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
)
from beam_search_utils import get_sector_mask

CHECKPOINT = (
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/'
    'results/pentagonbox_8_5_v3/work/results/'
    'straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint'
)
TOPOLOGY = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

t = Topology.from_dir(TOPOLOGY)
ibp_env.init_from_topology(t)
set_prime(1009)
env = IBPEnvironment()

with open(CHECKPOINT, 'rb') as f:
    ck = pickle.load(f)
print(f'Loaded checkpoint at step {ck["step"]}, beam size {len(ck["beam"])}')

s0 = ck['beam'][0]
print(f'state[0]: |expr|={len(s0.expr)}, |subs|={len(s0.subs)}, |resolved_subs|={len(s0.resolved_subs)}')

INTEGRAL = (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)
target_sector = tuple(get_sector_mask(INTEGRAL))
fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)
print(f'filtered: |fsubs|={len(fsubs)}, |fresolved|={len(fresolved)}')

# Step 1: full rebuild at state[0]
print('\n[1] Full rebuild at state[0] subs...')
t0 = time.time()
result_full_n, aux_n = compute_indirect_substituted_with_aux(
    fsubs, fresolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'  done in {time.time()-t0:.2f}s, |result|={len(result_full_n)}')

# Step 2: simulate adding ONE sub (pick the max-weight integral in target sector
# from state[0]'s expr as the target).
def max_weight_in_target_sector(expr):
    from sailir.ibp_env import weight as W, is_master
    cand = [(W(k)[:2], k) for k, v in expr.items() if v != 0 and not is_master(k)
            and all((target_sector[i] == 1) == (k[i] > 0) for i in range(8))]
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]

target = max_weight_in_target_sector(s0.expr)
print(f'\nPicked target: {target}')

# Construct a fake IBP equation for target (use first valid action from the indirect_cache).
# Easier: just pick any sub_int from fsubs and construct sol from its IBP equation.
# Actually, simpler: directly fabricate a fake sub for testing.
#
# For correctness testing, we need a valid (target, sol) pair where sol does not
# contain target. Let's pick target -> {some_other_integral: 1}.
# But that may not actually be a valid IBP-derived sub. For path-identity it
# doesn't matter — we just need to verify update logic.
#
# Use one of the existing subs from fsubs to test: pretend target -> fsubs[target']
# for some sub already in fsubs.
test_targets = [t for t in fsubs.keys() if t != target]
if not test_targets:
    print('ERROR: no test targets available'); sys.exit(1)
fake_target_for_test = (target[0]+1,) + target[1:]  # some integral NOT in subs
fake_sol = dict(list(fresolved.values())[0])  # any existing resolved value

print(f'Fake new sub: {fake_target_for_test} -> {len(fake_sol)} terms')

# new_resolved_subs and new_resolved_sol
new_resolved_sol = apply_resolved_subs(fake_sol, fresolved)
new_resolved = add_sub_to_resolved(fresolved, fake_target_for_test, fake_sol)
new_subs = dict(fsubs); new_subs[fake_target_for_test] = fake_sol

# Step 3: full rebuild at state[N+1]
print('\n[3] Full rebuild at NEW state (with extra sub)...')
t0 = time.time()
result_full_np1, aux_np1 = compute_indirect_substituted_with_aux(
    new_subs, new_resolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'  done in {time.time()-t0:.2f}s, |result|={len(result_full_np1)}')

# Step 4: incremental from aux_n
print('\n[4] Incremental update from aux_n + new sub...')
t0 = time.time()
result_inc, aux_inc = compute_indirect_substituted_incremental(
    aux_n, fake_target_for_test, new_resolved_sol, new_resolved,
    env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'  done in {time.time()-t0:.2f}s, |result|={len(result_inc)}')

# Step 5: compare full_np1 vs inc
print('\n[5] Comparing full rebuild vs incremental...')
if len(result_full_np1) != len(result_inc):
    print(f'  LENGTH MISMATCH: full={len(result_full_np1)}, inc={len(result_inc)}')
    sys.exit(1)

# Sort both by (sub_int, ibp_op, shift) for comparison
def key(entry):
    sub_int, ibp_op, shift, raw, cached, ub = entry
    return (sub_int, ibp_op, shift)

rfull = sorted(result_full_np1, key=key)
rinc  = sorted(result_inc,      key=key)

mismatches = 0
for (sf, opf, shf, rf, cf, uf), (si, opi, shi, ri, ci, ui) in zip(rfull, rinc):
    assert sf == si and opf == opi and shf == shi
    if cf != ci or uf != ui:
        mismatches += 1
        if mismatches <= 3:
            print(f'  MISMATCH on ({sf}, {opf}, {shf}):')
            print(f'    cached_full={cf}')
            print(f'    cached_inc ={ci}')
            print(f'    ub_full={uf}, ub_inc={ui}')

print(f'\nTotal entries: {len(rfull)}, mismatches: {mismatches}')
if mismatches == 0:
    print('SUCCESS: incremental == full rebuild (bit-identical)')
else:
    print(f'FAILURE: {mismatches} entries differ')
