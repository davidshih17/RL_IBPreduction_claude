#!/usr/bin/env python
"""For every survivor in A's beam at step K (matched by path with B's
beam survivor), enumerate valid actions for every tied target on BOTH
sides using production's enumerate_valid_actions_with_indirect_cache.
Report:
  - per (survivor, target): |valid_A|, |valid_B|, set-equal, order-equal
  - total: any SET mismatches \u2192 not pure-fp-drift cause
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


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def build_iraws(flat, rs, env):
    n_idx = ibp_env.N_INDICES
    iraws = []
    for row in flat.iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        iraws.append((sub_int, op, shift, raw, cached, ub))
    return iraws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_a')
    p.add_argument('ckpt_b')
    p.add_argument('topology')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    ca = load(args.ckpt_a)
    cb = load(args.ckpt_b)
    target_sector = tuple(ca['target_sector'])
    print(f'Comparing at step {ca["step"]} (B step={cb["step"]})')

    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}
    common = sorted(set(by_a) & set(by_b))
    print(f'Common-path survivors: {len(common)}/{len(by_a)}')

    n_set_mismatch = 0
    n_order_mismatch = 0
    n_total_tasks = 0
    set_mismatch_examples = []

    for s_idx, p_ in enumerate(common):
        sa = by_a[p_]
        sb = by_b[p_]
        non_masters = get_non_masters(sa['expr'], target_sector)
        if not non_masters:
            continue
        mw = tuple(sa['max_w'])
        tied = [k for k in non_masters if (weight(k)[0], weight(k)[1]) == mw]
        if not tied:
            continue

        iraws_a = build_iraws(sa['aux_flat'], sa['resolved_subs'], env)
        iraws_b = build_iraws(sb['aux_flat'], sb['resolved_subs'], env)

        for target in tied:
            n_total_tasks += 1
            valid_a = enumerate_valid_actions_with_indirect_cache(
                target, iraws_a, sa['subs'], sa['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            valid_b = enumerate_valid_actions_with_indirect_cache(
                target, iraws_b, sb['subs'], sb['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            set_a = set(valid_a)
            set_b = set(valid_b)
            set_eq = (set_a == set_b)
            order_eq = (valid_a == valid_b)
            if not set_eq:
                n_set_mismatch += 1
                only_a = set_a - set_b
                only_b = set_b - set_a
                set_mismatch_examples.append({
                    'rank': s_idx, 'target': target,
                    'sizes': (len(valid_a), len(valid_b)),
                    'A_only': len(only_a), 'B_only': len(only_b),
                    'sample_A_only': next(iter(only_a)) if only_a else None,
                    'sample_B_only': next(iter(only_b)) if only_b else None,
                })
            elif not order_eq:
                n_order_mismatch += 1

    print(f'\nTotal (survivor, target) tasks: {n_total_tasks}')
    print(f'  Valid SET mismatches:   {n_set_mismatch}')
    print(f'  Valid ORDER mismatches: {n_order_mismatch} '
          f'(set-equal but order-differs)')
    print(f'  Bit-identical valid:    '
          f'{n_total_tasks - n_set_mismatch - n_order_mismatch}')

    if set_mismatch_examples:
        print(f'\n=== SET MISMATCH EXAMPLES (first 3) ===')
        for ex in set_mismatch_examples[:3]:
            print(f'  rank={ex["rank"]} target={ex["target"]}')
            print(f'    sizes A/B = {ex["sizes"]}')
            print(f'    A-only={ex["A_only"]}, B-only={ex["B_only"]}')
            print(f'    sample A-only = {ex["sample_A_only"]}')
            print(f'    sample B-only = {ex["sample_B_only"]}')
        print(f'\n*** v4-rescue and baseline have GENUINE action SET '
              f'differences \u2014 not just fp drift ***')
    else:
        print(f'\n*** All valid SETS bit-identical \u2014 confirms fp drift '
              f'is the SOLE cause of beam swap ***')
    return 0


if __name__ == '__main__':
    sys.exit(main())
