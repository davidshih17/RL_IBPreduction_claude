#!/usr/bin/env python3
"""Print exact key differences between inc and full for one mismatch."""
import sys, pickle
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    compute_indirect_substituted_with_aux,
    compute_indirect_substituted_incremental,
    apply_resolved_subs, apply_resolved_subs_batch,
    add_sub_to_resolved,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
)
from beam_search_utils import get_sector_mask

t = Topology.from_dir('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox')
ibp_env.init_from_topology(t)
set_prime(1009)
env = IBPEnvironment()
with open('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v3/work/results/straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint','rb') as f:
    ck = pickle.load(f)
s0 = ck['beam'][0]
target_sector = tuple(get_sector_mask((-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)))
fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)

fake_target = (1, 2, 1, 0, 2, 2, 1, 1, -2, -1, 0)
fake_sol = dict(list(fresolved.values())[0])
new_resolved_sol = apply_resolved_subs(fake_sol, fresolved)
new_resolved = add_sub_to_resolved(fresolved, fake_target, fake_sol)
new_subs = dict(fsubs); new_subs[fake_target] = fake_sol

# Compute aux_n
res_n, aux_n = compute_indirect_substituted_with_aux(
    fsubs, fresolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
# Full rebuild
res_full, _ = compute_indirect_substituted_with_aux(
    new_subs, new_resolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
# Incremental
res_inc, _ = compute_indirect_substituted_incremental(
    aux_n, fake_target, new_resolved_sol, new_resolved,
    env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)

# Pick the entry with the mismatch
k_target = ((1, 2, 1, 0, 2, 2, 1, 1, -2, -1, 0), 9, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
ef = next(e for e in res_full if (e[0], e[1], e[2]) == k_target)
ei = next(e for e in res_inc if (e[0], e[1], e[2]) == k_target)

print(f'Same raw object: {ef[3] is ei[3]}')
raw = ef[3]
direct = apply_resolved_subs_batch([raw], new_resolved)[0]

print(f'|raw|={len(raw)}, raw fake_target in: {fake_target in raw}')
print(f'|direct apply(raw,new_resolved)|={len(direct)}')
print(f'|full cached|={len(ef[4])}')
print(f'|inc  cached|={len(ei[4])}')

# Keys in direct but not in inc
keys_in_full_not_inc = set(ef[4].keys()) - set(ei[4].keys())
keys_in_inc_not_full = set(ei[4].keys()) - set(ef[4].keys())
print(f'\nkeys in full not in inc ({len(keys_in_full_not_inc)}):')
for k in sorted(keys_in_full_not_inc):
    print(f'  {k}: full={ef[4][k]}')

print(f'\nkeys in inc not in full ({len(keys_in_inc_not_full)}):')
for k in sorted(keys_in_inc_not_full):
    print(f'  {k}: inc={ei[4][k]}')

# What does aux_n's cached look like for this raw? It should be apply(raw, OLD).
# Find which idx in aux_n's cached_unique corresponds to this raw.
prev_cu, prev_ubm, prev_rid, prev_iraws = aux_n
print(f'\nid(raw) in prev_rid? {id(raw) in prev_rid}')
if id(raw) in prev_rid:
    idx = prev_rid[id(raw)]
    old_c = prev_cu[idx]
    print(f'|old_c|={len(old_c)}, fake_target in old_c: {fake_target in old_c}')
    if fake_target in old_c:
        print(f'  old_c[fake_target] = {old_c[fake_target]}')
    # Verify with direct apply(raw, OLD)
    direct_old = apply_resolved_subs_batch([raw], fresolved)[0]
    print(f'|direct apply(raw,fresolved)|={len(direct_old)}')
    print(f'old_c == direct_old: {old_c == direct_old}')
    if old_c == direct_old:
        print(f'  fake_target in direct_old: {fake_target in direct_old}')
    print(f'inc cached == old_c (no Phase A change)? {ei[4] == old_c}')
