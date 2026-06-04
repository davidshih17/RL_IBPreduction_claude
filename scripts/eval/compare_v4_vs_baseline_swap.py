#!/usr/bin/env python
"""Compare v4-rescue and baseline at the step 166->167 transition.
For each step-167 survivor in each run, identify (parent_idx_in_step166,
picked_action). Classify the picked action as Phase 1a vs 1b. Find which
survivors differ (the 4 swap candidates). For swap candidates, determine
whether the v4 pick is Phase 1a or 1b, and locate the action's idx in
each run's valid list (to see if it sits beyond max_actions=900 in the
other run).

Also compare cumulative score of each step-166 survivor between runs.
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


def build_iraws(sp, env):
    n_idx = ibp_env.N_INDICES
    rs = sp['resolved_subs']
    out = []
    for row in sp['aux_flat'].iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        out.append((sub_int, op, shift, raw, cached, ub))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-dir', required=True)
    p.add_argument('--v4-dir', required=True)
    p.add_argument('--topology', required=True)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    with open(f'{args.baseline_dir}/result.pkl.ckpt.r1.step0166', 'rb') as f:
        bp = pickle.load(f)
    with open(f'{args.baseline_dir}/result.pkl.ckpt.r1.step0167', 'rb') as f:
        bc = pickle.load(f)
    with open(f'{args.v4_dir}/result.pkl.ckpt.r1.step0166', 'rb') as f:
        vp = pickle.load(f)
    with open(f'{args.v4_dir}/result.pkl.ckpt.r1.step0167', 'rb') as f:
        vc = pickle.load(f)
    ts = tuple(bp['target_sector'])

    # Step-166 score comparison: same path-key -> same state?
    by_b_p = {pk(s): (i, s) for i, s in enumerate(bp['beam'])}
    by_v_p = {pk(s): (i, s) for i, s in enumerate(vp['beam'])}
    common_paths = sorted(set(by_b_p) & set(by_v_p))
    print(f'Step 166: baseline {len(bp["beam"])}, v4 {len(vp["beam"])}, common-path {len(common_paths)}')
    score_drift = []
    for p_ in common_paths:
        sb = by_b_p[p_][1]
        sv = by_v_p[p_][1]
        if sb['expr'] == sv['expr'] and sb['subs'] == sv['subs']:
            score_drift.append((p_, sb['score'], sv['score'], sv['score'] - sb['score']))
    if score_drift:
        deltas = [d[3] for d in score_drift]
        print(f'Step-166 score drift over {len(score_drift)} matching states:')
        print(f'  max |Δ|={max(abs(d) for d in deltas):.6f}  '
              f'mean Δ={sum(deltas)/len(deltas):+.6f}  '
              f'max +Δ={max(deltas):+.6f}  min -Δ={min(deltas):+.6f}')
    # show 5 with largest |Δ|
    score_drift.sort(key=lambda x: -abs(x[3]))
    for sd in score_drift[:5]:
        print(f'    Δscore={sd[3]:+.5f}  baseline={sd[1]:.5f}  v4={sd[2]:.5f}')

    # Step-167 child compare: classify each child pick
    by_b_c = {pk(s): (i, s) for i, s in enumerate(bc['beam'])}
    by_v_c = {pk(s): (i, s) for i, s in enumerate(vc['beam'])}
    paths_b = set(by_b_c)
    paths_v = set(by_v_c)
    both = paths_b & paths_v
    only_b = paths_b - paths_v
    only_v = paths_v - paths_b
    print(f'\nStep 167: both={len(both)}, only-baseline={len(only_b)}, only-v4={len(only_v)}')

    def classify(parent_path, picked, src_p_idx_map, src_p_list, label):
        if parent_path not in src_p_idx_map:
            return None
        ip, sp = src_p_idx_map[parent_path]
        nm = get_non_masters(sp['expr'], ts)
        mw = tuple(sp['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        iraws = build_iraws(sp, env)
        rs = sp['resolved_subs']
        n_idx = ibp_env.N_INDICES
        for target in tied:
            valid = enumerate_valid_actions_with_indirect_cache(
                target, iraws, sp['subs'], rs,
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            for idx, (op, delta) in enumerate(valid):
                if (picked[1] == op and tuple(picked[2]) == tuple(delta)
                        and tuple(picked[0]) == tuple(target)):
                    seed = tuple(target[i] + delta[i] for i in range(n_idx))
                    raw = env.get_raw_equation_cached(op, seed)
                    is_p1a = (target in raw and raw[target] != 0)
                    return {'parent_idx': ip, 'target': target, 'op': op,
                            'delta': delta, 'idx_in_valid': idx,
                            'len_valid': len(valid), 'is_p1a': is_p1a}
        return None

    print(f'\n--- Baseline-only step-167 survivors (will be unique to baseline) ---')
    for p_ in sorted(only_b)[:10]:
        parent_path = p_[:-1]
        picked = p_[-1]
        cls_b = classify(parent_path, picked, by_b_p, bp['beam'], 'b')
        if cls_b is None:
            print(f'  parent not in baseline step-166')
            continue
        # also check where this same action sits in v4's first 900
        cls_v_same_parent = classify(parent_path, picked, by_v_p, vp['beam'], 'v')
        print(f'  parent={cls_b["parent_idx"]}  target={cls_b["target"]}  '
              f'op={cls_b["op"]} delta={cls_b["delta"]}')
        print(f'    baseline: idx={cls_b["idx_in_valid"]}/{cls_b["len_valid"]} '
              f'is_p1a={cls_b["is_p1a"]}  within900={cls_b["idx_in_valid"]<900}')
        if cls_v_same_parent:
            print(f'    v4 same act: idx={cls_v_same_parent["idx_in_valid"]}/'
                  f'{cls_v_same_parent["len_valid"]} '
                  f'within900={cls_v_same_parent["idx_in_valid"]<900}')

    print(f'\n--- V4-only step-167 survivors (unique to v4) ---')
    for p_ in sorted(only_v)[:10]:
        parent_path = p_[:-1]
        picked = p_[-1]
        cls_v = classify(parent_path, picked, by_v_p, vp['beam'], 'v')
        if cls_v is None:
            print(f'  parent not in v4 step-166')
            continue
        cls_b_same = classify(parent_path, picked, by_b_p, bp['beam'], 'b')
        print(f'  parent={cls_v["parent_idx"]}  target={cls_v["target"]}  '
              f'op={cls_v["op"]} delta={cls_v["delta"]}')
        print(f'    v4: idx={cls_v["idx_in_valid"]}/{cls_v["len_valid"]} '
              f'is_p1a={cls_v["is_p1a"]}  within900={cls_v["idx_in_valid"]<900}')
        if cls_b_same:
            print(f'    baseline same act: idx={cls_b_same["idx_in_valid"]}/'
                  f'{cls_b_same["len_valid"]} '
                  f'within900={cls_b_same["idx_in_valid"]<900}')


if __name__ == '__main__':
    sys.exit(main())
