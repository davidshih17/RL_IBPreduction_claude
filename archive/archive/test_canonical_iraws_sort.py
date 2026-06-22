#!/usr/bin/env python
"""Test the proposed fix: sort v4-rescue's iraws by canonical key
(sub_int_index_in_RS_insertion_order, op, shift_index_in_topology).
Then compare valid lists (set + order) to baseline's valid for the same
(survivor, target) at step K.

If sorted-v4's valid == baseline's valid (set AND order) for all 160
(survivor, target) tasks, the canonical-sort fix produces bit-identical
valid lists \u2014 which would eliminate the truncation-induced score drift.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, weight,
)
from beam_search_utils import get_non_masters


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_iraws_from_meta(iraws_meta, rs, env):
    """Build full iraws list of (sub_int, op, shift, raw, cached, union_bm)
    from FlatAux's iraws_meta numpy array."""
    n_idx = ibp_env.N_INDICES
    iraws = []
    for row in iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        iraws.append((sub_int, op, shift, raw, cached, ub))
    return iraws


def canonical_sort_key(env, rs_index, sub_int, op, shift):
    """Sort key matching baseline's iraws iteration order:
        (sub_int_position_in_RS_insertion_order, op_id, shift_position_in_topology)
    """
    key_idx = rs_index.get(sub_int, 10**9)  # unknown subs sink to end
    shift_list = env.shifts.get(op, [])
    try:
        shift_idx = shift_list.index(shift)
    except ValueError:
        shift_idx = 10**9
    return (key_idx, op, shift_idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_v4')
    p.add_argument('ckpt_base')
    p.add_argument('topology')
    p.add_argument('--max-survivors', type=int, default=40)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    ca = load(args.ckpt_v4)
    cb = load(args.ckpt_base)
    target_sector = tuple(ca['target_sector'])
    print(f'v4 step={ca["step"]} base step={cb["step"]}')

    def pk(s): return tuple(tuple(a) for a in s['path'])
    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}
    common = sorted(set(by_a) & set(by_b))
    print(f'Common-path survivors: {len(common)}')

    n_tasks = 0
    n_unsorted_setEq = 0
    n_unsorted_orderEq = 0
    n_sorted_setEq = 0
    n_sorted_orderEq = 0
    first_mismatch_after_sort = None

    for s_idx, p_ in enumerate(common[:args.max_survivors]):
        sa = by_a[p_]
        sb = by_b[p_]
        non_masters_a = get_non_masters(sa['expr'], target_sector)
        if not non_masters_a:
            continue
        mw = tuple(sa['max_w'])
        tied = [k for k in non_masters_a if (weight(k)[0], weight(k)[1]) == mw]
        if not tied:
            continue

        # RS insertion-order indices for v4's RS (which should equal baseline's RS at this step since both bit-identical state-wise)
        rs_index_a = {k: i for i, k in enumerate(sa['resolved_subs'].keys())}
        rs_index_b = {k: i for i, k in enumerate(sb['resolved_subs'].keys())}

        # Original iraws
        iraws_a_orig = build_iraws_from_meta(sa['aux_flat'].iraws_meta, sa['resolved_subs'], env)
        iraws_b = build_iraws_from_meta(sb['aux_flat'].iraws_meta, sb['resolved_subs'], env)

        # Sorted iraws via canonical key
        iraws_a_sorted = sorted(
            iraws_a_orig,
            key=lambda e: canonical_sort_key(env, rs_index_a, e[0], e[1], e[2]),
        )
        # Also sort baseline by same canonical key (should be a no-op if baseline is already canonical)
        iraws_b_sorted = sorted(
            iraws_b,
            key=lambda e: canonical_sort_key(env, rs_index_b, e[0], e[1], e[2]),
        )
        baseline_already_canonical = (iraws_b == iraws_b_sorted)

        for target in tied:
            n_tasks += 1
            va_unsorted = enumerate_valid_actions_with_indirect_cache(
                target, iraws_a_orig, sa['subs'], sa['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            va_sorted = enumerate_valid_actions_with_indirect_cache(
                target, iraws_a_sorted, sa['subs'], sa['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            vb = enumerate_valid_actions_with_indirect_cache(
                target, iraws_b, sb['subs'], sb['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            if set(va_unsorted) == set(vb):
                n_unsorted_setEq += 1
            if va_unsorted == vb:
                n_unsorted_orderEq += 1
            if set(va_sorted) == set(vb):
                n_sorted_setEq += 1
            if va_sorted == vb:
                n_sorted_orderEq += 1
            else:
                if first_mismatch_after_sort is None:
                    first_mismatch_after_sort = (s_idx, target, va_sorted, vb)

    print(f'\nResults across {n_tasks} (survivor, target) tasks:')
    print(f'  UNSORTED v4 vs baseline:  set-eq {n_unsorted_setEq}/{n_tasks},  '
          f'order-eq {n_unsorted_orderEq}/{n_tasks}')
    print(f'  SORTED   v4 vs baseline:  set-eq {n_sorted_setEq}/{n_tasks},  '
          f'order-eq {n_sorted_orderEq}/{n_tasks}')
    print(f'  Baseline iraws already canonical: {baseline_already_canonical}')

    if first_mismatch_after_sort is not None and n_sorted_orderEq < n_tasks:
        s_idx, target, va, vb = first_mismatch_after_sort
        # find first position differing
        for i in range(min(len(va), len(vb))):
            if va[i] != vb[i]:
                print(f'\nFirst sorted-v4 vs baseline mismatch at survivor #{s_idx}, '
                      f'target={target}, position {i}:')
                print(f'  v4: {va[i]}')
                print(f'  base: {vb[i]}')
                # show surrounding 3 positions
                for j in range(max(0, i-2), min(len(va), len(vb), i+3)):
                    print(f'    [{j}] v4={va[j]}  base={vb[j]}')
                break
    return 0


if __name__ == '__main__':
    sys.exit(main())
