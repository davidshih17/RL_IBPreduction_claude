#!/usr/bin/env python
"""For the step 166->167 transition on baseline beam[0], find the index of
the picked action in the valid list. Tells us whether the picked action sat
within the first 900 (so truncation cap didn't matter) or beyond.
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
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, weight,
)
from beam_search_utils import get_non_masters


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt166')
    p.add_argument('ckpt167')
    p.add_argument('topology')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    with open(args.ckpt166, 'rb') as f:
        c166 = pickle.load(f)
    with open(args.ckpt167, 'rb') as f:
        c167 = pickle.load(f)
    ts = tuple(c166['target_sector'])

    # Identify each step-167 survivor's parent (= step-166 survivor whose
    # path is a prefix), and the action that produced it.
    by_path166 = {pk(s): (i, s) for i, s in enumerate(c166['beam'])}
    n_idx = ibp_env.N_INDICES

    print(f'Step 166 beam: {len(c166["beam"])}  Step 167 beam: {len(c167["beam"])}\n')

    # For each 167-survivor, find parent + picked action + its idx in parent's valid
    examined = 0
    for j, sc in enumerate(c167['beam'][:10]):  # first 10 children
        path = pk(sc)
        parent_path = path[:-1]
        picked = path[-1]
        if parent_path not in by_path166:
            print(f'child {j}: parent not in step-166 beam (rare); skip')
            continue
        i_par, sp = by_path166[parent_path]
        nm = get_non_masters(sp['expr'], ts)
        mw = tuple(sp['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        # Build iraws for parent and enumerate valid for each tied target
        iraws = []
        rs = sp['resolved_subs']
        for row in sp['aux_flat'].iraws_meta:
            sub_int = tuple(int(x) for x in row[:n_idx])
            op = int(row[n_idx])
            shift = tuple(int(x) for x in row[n_idx + 1:])
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(op, seed)
            cached = apply_resolved_subs(raw, rs)
            ub = cached_union_bitmask(cached)
            iraws.append((sub_int, op, shift, raw, cached, ub))

        found = False
        for target in tied:
            valid = enumerate_valid_actions_with_indirect_cache(
                target, iraws, sp['subs'], rs,
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            for idx, (op, delta) in enumerate(valid):
                # picked = (sub_int_or_target, op, shift)
                # Production action format = (target, op, delta) per path
                if picked[1] == op and tuple(picked[2]) == tuple(delta) and \
                   tuple(picked[0]) == tuple(target):
                    print(f'child {j}: parent_idx={i_par} target={target}  '
                          f'op={op} delta={delta}')
                    print(f'  picked action sits at idx {idx}/{len(valid)} in '
                          f'parent\'s valid (mw={mw}, |tied|={len(tied)})')
                    print(f'  within first 900? {idx < 900}'
                          f'  within first 4000? {idx < 4000}')
                    found = True
                    break
            if found:
                break
        if not found:
            print(f'child {j}: picked action not found in any tied valid')
        examined += 1

    print(f'\nExamined {examined} children of step 166 beam.')


if __name__ == '__main__':
    sys.exit(main())
