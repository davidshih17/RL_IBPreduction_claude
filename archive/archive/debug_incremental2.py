#!/usr/bin/env python3
"""Compare incremental vs full rebuild on a SINGLE entry — print where they diverge."""
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
    add_sub_to_resolved, get_raw_equation,
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
INTEGRAL = (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)
target_sector = tuple(get_sector_mask(INTEGRAL))
fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)

fake_target = (1, 2, 1, 0, 2, 2, 1, 1, -2, -1, 0)
fake_sol = dict(list(fresolved.values())[0])
new_resolved_sol = apply_resolved_subs(fake_sol, fresolved)
new_resolved = add_sub_to_resolved(fresolved, fake_target, fake_sol)
new_subs = dict(fsubs); new_subs[fake_target] = fake_sol

# Run BOTH compute functions
res_full, aux_full = compute_indirect_substituted_with_aux(
    new_subs, new_resolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'full: |result|={len(res_full)}, |cached_unique|={len(aux_full[0])}')

res_n, aux_n = compute_indirect_substituted_with_aux(
    fsubs, fresolved, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'compute@N: |result|={len(res_n)}, |cached_unique|={len(aux_n[0])}')

res_inc, aux_inc = compute_indirect_substituted_incremental(
    aux_n, fake_target, new_resolved_sol, new_resolved,
    env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache)
print(f'incremental: |result|={len(res_inc)}, |cached_unique|={len(aux_inc[0])}')

# Find one mismatching entry. Compare for sub_int=fake_target specifically.
print('\n=== Sample mismatches ===')
def key(e):
    return (e[0], e[1], e[2])
import collections
full_by_key = {key(e): e for e in res_full}
inc_by_key  = {key(e): e for e in res_inc}
common = set(full_by_key) & set(inc_by_key)
mm = 0
for k in common:
    if full_by_key[k][4] != inc_by_key[k][4]:
        mm += 1
        if mm <= 1:
            sub_int, op, sh = k
            ef, ei = full_by_key[k], inc_by_key[k]
            print(f'MISMATCH on sub_int={sub_int}, op={op}, shift={sh}')
            print(f'  raw id full={id(ef[3])}, raw id inc={id(ei[3])}, same? {ef[3] is ei[3]}')
            # The raws ARE the same object (raw_eq_cache returns same dict)
            print(f'  raw keys: {sorted(ef[3].keys())[:3]}')
            # Apply new_resolved directly to raw
            direct = apply_resolved_subs_batch([ef[3]], new_resolved)[0]
            print(f'  direct apply(raw,new_resolved): |{len(direct)}|, first keys: {sorted(direct.keys())[:3]}')
            print(f'  full cached:   |{len(ef[4])}|, first keys: {sorted(ef[4].keys())[:3]}')
            print(f'  inc  cached:   |{len(ei[4])}|, first keys: {sorted(ei[4].keys())[:3]}')
            print(f'  direct == full: {direct == ef[4]}')
            print(f'  direct == inc:  {direct == ei[4]}')
            # Which sub_int's cached is inc_cached identical to?
            for idx_, c_ in enumerate(aux_n[0]):
                if c_ == ei[4]:
                    print(f'  inc cached == aux_n.cached_unique[{idx_}] (i.e. an OLD cached at index {idx_})')
                    break
print(f'\ntotal mismatches across {len(common)} common keys: {mm}')
