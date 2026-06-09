#!/usr/bin/env python
"""Use production's enumerate_valid_actions_with_indirect_cache directly
on the reconstructed parent state. Also runs the model to compare logits/
action_probs between A and B.
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
)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def reconstruct_iraws(flat, env):
    n_idx = (flat.iraws_meta.shape[1] - 1) // 2
    iraws = []
    for row in flat.iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, env.__dict__.get('_tmp_rs', {}))
        # caller fills in cached per-survivor
        iraws.append([sub_int, op, shift, raw, None, None])
    return iraws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_a')
    p.add_argument('ckpt_b')
    p.add_argument('topology')
    p.add_argument('--target-path-step', type=int, default=26)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    ca = load(args.ckpt_a)
    cb = load(args.ckpt_b)

    # Iterate all matched survivors and find ones with valid-list diffs.

    target_sector = tuple(ca['target_sector'])
    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}

    # Use production enumerate on both sides for the same (state, target) pair.
    from beam_search_utils import get_non_masters

    for path in by_a:
        if path not in by_b:
            continue
        sa = by_a[path]
        sb = by_b[path]
        non_masters_a = get_non_masters(sa['expr'], target_sector)
        if not non_masters_a:
            continue
        # Pick tied max-w target
        from sailir.ibp_env import weight
        mw = tuple(sa['max_w'])
        tied = [k for k in non_masters_a if (weight(k)[0], weight(k)[1]) == mw]
        if not tied:
            continue
        target = tied[0]

        # Build iraws as production list with cached.
        flat_a = sa['aux_flat']
        flat_b = sb['aux_flat']
        rs_a = sa['resolved_subs']
        rs_b = sb['resolved_subs']

        iraws_a = []
        for row in flat_a.iraws_meta:
            n_idx = ibp_env.N_INDICES
            sub_int = tuple(int(x) for x in row[:n_idx])
            op = int(row[n_idx])
            shift = tuple(int(x) for x in row[n_idx + 1:])
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(op, seed)
            cached = apply_resolved_subs(raw, rs_a)
            from sailir.ibp_env import cached_union_bitmask
            ub = cached_union_bitmask(cached)
            iraws_a.append((sub_int, op, shift, raw, cached, ub))

        iraws_b = []
        for row in flat_b.iraws_meta:
            n_idx = ibp_env.N_INDICES
            sub_int = tuple(int(x) for x in row[:n_idx])
            op = int(row[n_idx])
            shift = tuple(int(x) for x in row[n_idx + 1:])
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(op, seed)
            cached = apply_resolved_subs(raw, rs_b)
            from sailir.ibp_env import cached_union_bitmask
            ub = cached_union_bitmask(cached)
            iraws_b.append((sub_int, op, shift, raw, cached, ub))

        subs_a = sa['subs']
        subs_b = sb['subs']

        valid_a = enumerate_valid_actions_with_indirect_cache(
            target, iraws_a, subs_a, rs_a, env.ibp_t, env.li_t, env.shifts,
            'subsector', env._raw_eq_cache,
        )
        valid_b = enumerate_valid_actions_with_indirect_cache(
            target, iraws_b, subs_b, rs_b, env.ibp_t, env.li_t, env.shifts,
            'subsector', env._raw_eq_cache,
        )

        print(f'\nSurvivor max_w={sa["max_w"]} nm={sa["n_non_masters"]} '
              f'target={target}')
        print(f'  |valid_A|={len(valid_a)}  |valid_B|={len(valid_b)}')

        # Set + order comparison
        set_a = set(valid_a)
        set_b = set(valid_b)
        if set_a != set_b:
            print(f'  Set diff: A-only={len(set_a - set_b)}, '
                  f'B-only={len(set_b - set_a)}')
            sa_only = list(set_a - set_b)[:2]
            sb_only = list(set_b - set_a)[:2]
            print(f'  A-only example: {sa_only}')
            print(f'  B-only example: {sb_only}')
        else:
            # Order check up to MAX
            MAX = 900
            order_diff = sum(1 for k in range(min(len(valid_a), len(valid_b), MAX))
                             if valid_a[k] != valid_b[k])
            print(f'  Sets match; first {MAX} order: {order_diff} positions differ')

        if len(valid_a) > 5:
            break
    return 0


if __name__ == '__main__':
    sys.exit(main())
