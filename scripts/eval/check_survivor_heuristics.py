#!/usr/bin/env python
"""For step K-1 -> K, examine which actions produced surviving children.
For each (parent, picked action) pair, compute features. Compare against
the distribution of features in the full valid list to find heuristics
that correlate with 'survivor-producing' actions.

The hypothesis: surviving actions may share structural features (small
delta, small raw, recent anchor sub_int, etc.) that the model partially
captures but not perfectly.
"""
import argparse
import pickle
import sys
import numpy as np

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


def featurize_action(target, op, delta, raw, cached, rs, rs_index, expr_nm_set):
    is_p1a = (target in raw and raw[target] != 0)
    anchor_recency = -1
    if not is_p1a:
        for k in raw:
            if k in rs and target in rs[k]:
                anchor_recency = rs_index.get(k, -1)
                break
    delta_norm = sum(abs(d) for d in delta)
    delta_max = max(abs(d) for d in delta) if delta else 0
    cached_in_expr_nm = sum(1 for k in cached if k in expr_nm_set)
    return {
        'is_p1a': is_p1a,
        'anchor_recency': anchor_recency,
        'raw_size': len(raw),
        'cached_size': len(cached),
        'cached_in_expr_nm': cached_in_expr_nm,
        'delta_norm': delta_norm,
        'delta_max': delta_max,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_parent')
    p.add_argument('ckpt_child')
    p.add_argument('topology')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    with open(args.ckpt_parent, 'rb') as f:
        cp = pickle.load(f)
    with open(args.ckpt_child, 'rb') as f:
        cc = pickle.load(f)
    ts = tuple(cp['target_sector'])
    n_idx = ibp_env.N_INDICES

    by_path_p = {pk(s): (i, s) for i, s in enumerate(cp['beam'])}

    # For each child survivor, identify (parent_idx, picked action). Then for
    # that parent, enumerate full valid, compute features per action, and
    # mark which idx the picked action sits at.

    # Collect picked-action features and pooled all-action features.
    picked_feats = []
    all_feats = []
    picked_idxs = []  # idx within parent's valid
    picked_actions = set()  # uniqueness: how many distinct picks across 40 children?
    # We'll also record full feature lists for one example parent
    sample_parent_dump = None

    for j, sc in enumerate(cc['beam']):
        path = pk(sc)
        parent_path = path[:-1]
        picked = path[-1]
        if parent_path not in by_path_p:
            continue
        i_par, sp = by_path_p[parent_path]
        nm = get_non_masters(sp['expr'], ts)
        mw = tuple(sp['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        rs = sp['resolved_subs']
        rs_keys = list(rs.keys())
        rs_index = {k: i for i, k in enumerate(rs_keys)}
        expr_nm_set = set(nm.keys())

        iraws = []
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
            # Feature pool for THIS target's valid
            parent_feats = []
            picked_idx_local = -1
            for idx, (op, delta) in enumerate(valid):
                seed = tuple(target[i] + delta[i] for i in range(n_idx))
                raw = env.get_raw_equation_cached(op, seed)
                cached = apply_resolved_subs(raw, rs)
                f = featurize_action(target, op, delta, raw, cached, rs,
                                     rs_index, expr_nm_set)
                f['idx'] = idx
                parent_feats.append(f)
                if (picked[1] == op and tuple(picked[2]) == tuple(delta)
                        and tuple(picked[0]) == tuple(target)):
                    picked_idx_local = idx
                    picked_feats.append(f)
                    picked_idxs.append(idx)
                    picked_actions.add((tuple(target), op, tuple(delta)))
                    found = True
            # Pool every action across all parents (for the matching target only)
            if picked_idx_local >= 0:
                all_feats.extend(parent_feats)
                if sample_parent_dump is None:
                    sample_parent_dump = (i_par, target, parent_feats, picked_idx_local)
            if found:
                break

    print(f'Child survivors examined: {len(picked_feats)}/{len(cc["beam"])}')
    print(f'Unique picked actions across survivors: {len(picked_actions)}')
    print(f'Picked-action idx distribution in parent valid:')
    pi = np.array(picked_idxs)
    if len(pi):
        print(f'  min={pi.min()}  max={pi.max()}  mean={pi.mean():.1f}  '
              f'p90={np.percentile(pi,90):.0f}  median={np.median(pi):.0f}')

    # Heuristic feature comparison
    keys = ('is_p1a', 'anchor_recency', 'raw_size', 'cached_size',
            'cached_in_expr_nm', 'delta_norm', 'delta_max')
    print(f'\nFeature stats: PICKED ({len(picked_feats)}) vs ALL valid ({len(all_feats)})')
    for k in keys:
        vp = np.array([f[k] for f in picked_feats], dtype=float)
        va = np.array([f[k] for f in all_feats], dtype=float)
        if k == 'is_p1a':
            print(f'  {k:>20s}: picked mean={vp.mean():.3f}  all mean={va.mean():.3f}')
        else:
            print(f'  {k:>20s}: picked mean={vp.mean():7.2f} std={vp.std():6.2f}  '
                  f'all mean={va.mean():7.2f} std={va.std():6.2f}  '
                  f'(picked median={np.median(vp):.0f}, all median={np.median(va):.0f})')

    # For each feature, percentile rank of picked actions in "all" distribution
    print(f'\nRank of picked actions within "all" distribution per feature '
          f'(lower percentile = picked is smaller than typical):')
    for k in keys:
        if k == 'is_p1a':
            continue
        va = np.array([f[k] for f in all_feats])
        vp = np.array([f[k] for f in picked_feats])
        # rank of each picked value within va
        ranks = []
        va_sorted = np.sort(va)
        for v in vp:
            r = np.searchsorted(va_sorted, v, side='right') / len(va_sorted)
            ranks.append(r)
        ranks = np.array(ranks)
        print(f'  {k:>20s}: picked-percentile median={np.median(ranks)*100:.1f}%  '
              f'mean={ranks.mean()*100:.1f}%  p25-p75=[{np.percentile(ranks,25)*100:.1f}, '
              f'{np.percentile(ranks,75)*100:.1f}]%')

    # Sample parent dump: show full ranking of valid by each candidate heuristic
    if sample_parent_dump is not None:
        i_par, target, pf, pidx = sample_parent_dump
        print(f'\n--- Sample parent {i_par} target={target} '
              f'(picked idx={pidx}/{len(pf)}) ---')
        for k in ('delta_norm', 'raw_size', 'cached_size', 'cached_in_expr_nm',
                 'anchor_recency'):
            vals = np.array([f[k] for f in pf], dtype=float)
            if vals.std() == 0:
                continue
            order = np.argsort(vals)
            ranks = np.argsort(order)
            r_picked = int(ranks[pidx])
            print(f'  sort by {k:>20s} ascending: picked sits at rank '
                  f'{r_picked}/{len(pf)} ({r_picked/len(pf)*100:.1f}%) '
                  f'value={pf[pidx][k]} (min={vals.min():.0f} max={vals.max():.0f})')


if __name__ == '__main__':
    sys.exit(main())
