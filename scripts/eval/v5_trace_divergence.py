#!/usr/bin/env python
"""Walk the v5 path step by step. At each step, maintain BOTH v5 state
(stripped) and full state (no stripping). At each step compute target's
coefficient in cached_v5 vs cached_full. If they differ on active terms,
we've found the bug.

For each step we also check: is the active part of expr_v5 == active part of expr_full?
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
    apply_substitution, apply_all_substitutions, apply_resolved_subs,
    solve_ibp_for, weight, is_master, PRIME,
)
from beam_search_v5 import (
    apply_action_v5, State_v5, is_active, strip_passenger,
    add_sub_to_resolved_v5, apply_substitution_v5,
)
from beam_search_utils import get_sector_mask


def active_target_part(d, target_sector, start_w12):
    """Restrict dict to active target-sector entries (non-master + master)."""
    return {k: v for k, v in d.items()
            if tuple(get_sector_mask(k)) == target_sector
            and is_active(k, start_w12)}


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
    # Prefer best_state.path; fall back to beam[0].path if best is empty
    path = d['best_state']['path']
    if not path and 'beam' in d:
        path = d['beam'][0]['path']
        print(f'best_state.path empty; using beam[0].path')
    print(f'Tracing {len(path)} steps')

    # State_v5
    state_v5 = State_v5(
        expr={start_int: 1},
        resolved_subs={},
        sub_accum={},
        score=0.0,
        path=[],
        n_non_masters=1,
    )
    # Full state (no stripping anywhere)
    expr_full = {start_int: 1}
    subs_full = {}

    n_idx = ibp_env.N_INDICES

    for step, (target, ibp_op, delta) in enumerate(path):
        seed = tuple(target[i] + delta[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(ibp_op, seed)

        # v5 cached
        cached_v5 = apply_resolved_subs(raw, state_v5.resolved_subs)
        # Full cached
        cached_full = apply_all_substitutions(raw, subs_full)

        coeff_target_v5 = cached_v5.get(target, 0)
        coeff_target_full = cached_full.get(target, 0)

        # Compare ACTIVE TARGET-SECTOR parts of cached_v5 and cached_full
        active_v5 = active_target_part(cached_v5, target_sector, start_w12)
        active_full = active_target_part(cached_full, target_sector, start_w12)
        equal = (active_v5 == active_full)

        # Also compare active(expr_v5) vs active(expr_full)
        expr_v5_active = active_target_part(state_v5.expr, target_sector, start_w12)
        expr_full_active = active_target_part(expr_full, target_sector, start_w12)
        expr_equal = (expr_v5_active == expr_full_active)

        # Detail line
        print(f'step {step+1:>2}: target_in_v5_cached={coeff_target_v5} '
              f'target_in_full_cached={coeff_target_full} '
              f'active_cached_equal={equal} '
              f'active_expr_equal={expr_equal}')

        if not equal:
            print(f'  ACTIVE CACHED DISAGREE!')
            only_v5 = set(active_v5) - set(active_full)
            only_full = set(active_full) - set(active_v5)
            disagree = {k for k in set(active_v5) & set(active_full)
                        if active_v5[k] != active_full[k]}
            print(f'    only in v5 cached: {len(only_v5)}')
            print(f'    only in full cached: {len(only_full)}')
            print(f'    different coeffs: {len(disagree)}')
            for k in list(disagree)[:5]:
                print(f'      {k}  v5={active_v5[k]}  full={active_full[k]}')
            for k in list(only_v5)[:3]:
                print(f'      v5-only {k}  coeff={active_v5[k]}')
            for k in list(only_full)[:3]:
                print(f'      full-only {k}  coeff={active_full[k]}')
            return 1

        if not expr_equal:
            print(f'  ACTIVE EXPR DISAGREE!')
            # Detail diff
            only_v5_e = set(expr_v5_active) - set(expr_full_active)
            only_full_e = set(expr_full_active) - set(expr_v5_active)
            disagree_e = {k for k in set(expr_v5_active) & set(expr_full_active)
                          if expr_v5_active[k] != expr_full_active[k]}
            print(f'    only in expr_v5: {len(only_v5_e)}')
            print(f'    only in expr_full: {len(only_full_e)}')
            print(f'    different coeffs: {len(disagree_e)}')
            for k in list(only_v5_e)[:5]:
                print(f'      v5-only: {k}  coeff={expr_v5_active[k]}')
            for k in list(only_full_e)[:5]:
                print(f'      full-only: {k}  coeff={expr_full_active[k]}  master={is_master(k)}')
            for k in list(disagree_e)[:5]:
                print(f'      diff: {k}  v5={expr_v5_active[k]}  full={expr_full_active[k]}')
            return 1

        # Both equal — proceed. But if target's coeff in cached differs, that's the bug.
        if coeff_target_v5 != coeff_target_full:
            print(f'  TARGET COEFF MISMATCH but active parts equal — INCONSISTENT (bug somewhere)')
            return 1

        if coeff_target_v5 == 0:
            print(f'  target absent in both — would fail solve. STOP.')
            return 1

        # Compute sols and advance state
        sol_v5 = solve_ibp_for(cached_v5, target)
        sol_full = solve_ibp_for(cached_full, target)


        # Update v5 state
        new_expr, new_sub_accum = apply_substitution_v5(
            state_v5.expr, state_v5.sub_accum, target, sol_v5,
            target_sector, start_w12,
        )
        new_rs = add_sub_to_resolved_v5(
            state_v5.resolved_subs, target, sol_v5, start_w12,
        )
        # Sanity check: every value should be < PRIME
        bad_expr = {k: v for k, v in new_expr.items() if v >= PRIME or v < 0}
        if bad_expr:
            print(f'  STEP {step+1}: expr has {len(bad_expr)} entries with un-reduced coeffs!')
            for k, v in list(bad_expr.items())[:5]:
                print(f'    {k}  v={v}  v%PRIME={v%PRIME}')
        bad_rs = {}
        for K, sol_K in new_rs.items():
            for k, v in sol_K.items():
                if v >= PRIME or v < 0:
                    bad_rs[(K, k)] = v
        if bad_rs:
            print(f'  STEP {step+1}: RS has {len(bad_rs)} un-reduced coeffs')
            for kk, v in list(bad_rs.items())[:5]:
                print(f'    {kk}  v={v}')
        state_v5 = State_v5(
            expr=new_expr, resolved_subs=new_rs,
            sub_accum=new_sub_accum,
            score=0.0, path=[], n_non_masters=0,
        )

        # Update full state
        subs_full[target] = sol_full
        expr_full = apply_substitution(expr_full, target, sol_full)

    print('Traced all 18 steps. No bug detected.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
