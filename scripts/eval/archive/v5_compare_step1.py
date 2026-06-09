#!/usr/bin/env python
"""At step 1 of the v5 path, RS is empty so v5 and full apply IDENTICAL
substitutions. Print the two resulting expr's side-by-side and check
where they disagree.

If active-target-sector parts disagree, there's a bug in apply_substitution_v5.
If they agree, the discrepancy is elsewhere.
"""
import argparse
import pickle
import sys

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_substitution, solve_ibp_for, weight, is_master, PRIME,
)
from beam_search_v5 import apply_action_v5, State_v5, is_active
from beam_search_utils import get_sector_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    start_int = tuple(int(x) for x in args.integral.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))

    with open(args.ckpt, 'rb') as f:
        d = pickle.load(f)
    path = d['best_state']['path']
    target, op, delta = path[0]
    print(f'Step 1 action: target={target} op={op} delta={delta}')
    print(f'start_w12={start_w12}  target_sector={target_sector}')

    # Compute sol the way replay does (full algebra, RS empty)
    seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
    raw = env.get_raw_equation_cached(op, seed)
    sol = solve_ibp_for(raw, target)
    print(f'\nsol has {len(sol)} entries')

    # Classify each entry of sol
    print('\nsol breakdown:')
    n_active_nm_target = 0
    n_active_master_target = 0
    n_pass_nm_target = 0
    n_pass_master_target = 0
    n_subsector = 0
    active_nm_target_entries = []
    for k, c in sol.items():
        sec = tuple(get_sector_mask(k))
        if sec != target_sector:
            n_subsector += 1
            continue
        w12 = (weight(k)[0], weight(k)[1])
        active = is_active(k, start_w12)
        master = is_master(k)
        if active and not master:
            n_active_nm_target += 1
            active_nm_target_entries.append((k, c, w12))
        elif active and master:
            n_active_master_target += 1
        elif not active and not master:
            n_pass_nm_target += 1
        elif not active and master:
            n_pass_master_target += 1
    print(f'  active target-sector non-master: {n_active_nm_target}')
    print(f'  active target-sector master:     {n_active_master_target}')
    print(f'  passenger target-sector non-master: {n_pass_nm_target}')
    print(f'  passenger target-sector master:  {n_pass_master_target}')
    print(f'  sub-sector:                       {n_subsector}')

    print(f'\nActive target-sector non-master entries in sol '
          f'(should all appear in expr_v5):')
    for k, c, w12 in active_nm_target_entries:
        print(f'  {k}  weight(w1,w2)={w12}  coeff={c}')

    # Apply via v5
    initial = State_v5(
        expr={start_int: 1},
        resolved_subs={},
        sub_accum={},
        score=0.0,
        path=[],
        n_non_masters=1,
    )
    child = apply_action_v5(initial, target, op, delta, 1.0, env, target_sector, start_w12)
    if child is None:
        print('\napply_action_v5 returned None! BUG')
        return 1
    print(f'\nexpr_v5 after step 1: {len(child.expr)} entries')
    for k, c in child.expr.items():
        w12 = (weight(k)[0], weight(k)[1])
        master = is_master(k)
        sec = tuple(get_sector_mask(k))
        print(f'  {k}  weight={w12}  in_target_sector={sec == target_sector}  '
              f'is_master={master}  coeff={c}')

    # Apply via full replay path (start_expr -> apply sub for start_int -> sol)
    full_expr = apply_substitution({start_int: 1}, target, sol)
    print(f'\nexpr_full after step 1: {len(full_expr)} entries (in target sector + active filter):')
    n_active_nm_in_full_expr = 0
    for k, c in full_expr.items():
        sec = tuple(get_sector_mask(k))
        if sec != target_sector:
            continue
        if is_master(k):
            continue
        w12 = (weight(k)[0], weight(k)[1])
        if w12 >= start_w12:
            print(f'  {k}  weight={w12}  coeff={c}')
            n_active_nm_in_full_expr += 1

    print(f'\nSummary: sol has {n_active_nm_target} active target-sector non-masters,')
    print(f'         expr_v5 has {len(child.expr)} entries,')
    print(f'         expr_full has {n_active_nm_in_full_expr} active target-sector non-masters.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
