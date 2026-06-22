#!/usr/bin/env python
"""For step-166 step-the-model-preferred top-K actions, dump features that
could be heuristics for action ranking. Check correlation between each
feature and model probability/rank.
"""
import argparse
import pickle
import sys
import torch
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
from beam_search_full import prepare_batched_input_v5
from beam_search_utils import get_non_masters
from sailir.classifier import IBPActionClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('topology')
    p.add_argument('--model', required=True)
    p.add_argument('--top-k', type=int, default=200)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    torch.set_num_threads(1)

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    with open(args.ckpt, 'rb') as f:
        c = pickle.load(f)
    ts = tuple(c['target_sector'])
    s = c['beam'][0]  # use first beam survivor
    nm = get_non_masters(s['expr'], ts)
    mw = tuple(s['max_w'])
    tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
    target = tied[0]
    print(f'Survivor: max_w={mw} target={target}')

    # Build iraws + cached + valid
    n_idx = ibp_env.N_INDICES
    rs = s['resolved_subs']
    rs_keys = list(rs.keys())  # insertion order = step order
    rs_index = {k: i for i, k in enumerate(rs_keys)}
    print(f'#RS keys (sub_ints in chronological order): {len(rs_keys)}')

    iraws_list = []  # we keep iraws_meta order; full struct for enumerate
    for row in s['aux_flat'].iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        iraws_list.append((sub_int, op, shift, raw, cached, ub))

    valid = enumerate_valid_actions_with_indirect_cache(
        target, iraws_list, s['subs'], rs,
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )

    # Find Phase 1a prefix length
    n_p1a = 0
    for op, delta in valid:
        seed = tuple(target[i] + delta[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        if target in raw and raw[target] != 0:
            n_p1a += 1
        else:
            break
    print(f'|valid|={len(valid)}  Phase 1a prefix length={n_p1a}')

    # Build feature table for each valid action
    expr_nm_set = set(nm.keys())
    print(f'|expr_nm|={len(expr_nm_set)}')

    # For each action: compute features
    # We want (idx, is_p1a, anchor_sub_int_recency, |raw|, |cached|, |cached \cap expr_nm|, delta_norm)
    feats = []
    for idx, (op, delta) in enumerate(valid):
        seed = tuple(target[i] + delta[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        is_p1a = (target in raw and raw[target] != 0)
        # Anchor recency: for Phase 1b, find the sub_int (= K in raw) that triggers the substitution.
        # For Phase 1a, anchor doesn't apply (action is direct).
        if is_p1a:
            anchor_recency = -1  # N/A
        else:
            # Find K in raw such that K \in rs and target \in sol_K
            anchor_recency = -1
            for k in raw:
                if k in rs and target in rs[k]:
                    anchor_recency = rs_index.get(k, -1)
                    break
        delta_norm = sum(abs(d) for d in delta)
        cached_in_expr_nm = sum(1 for k in cached if k in expr_nm_set)
        feats.append({
            'idx': idx, 'op': op, 'delta': delta, 'is_p1a': is_p1a,
            'anchor_recency': anchor_recency,
            'raw_size': len(raw),
            'cached_size': len(cached),
            'cached_in_expr_nm': cached_in_expr_nm,
            'delta_norm': delta_norm,
        })

    # Run model
    bd = [(s['expr'], s['subs'], valid, ts, target)]
    b, _ = prepare_batched_input_v5(bd, 'cpu')
    with torch.no_grad():
        logits, probs = model(
            b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
            b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
            b['sub_repl_mask'], b['sub_mask'],
            b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
            b['sector_mask'], b['target_integral'],
        )
    n_v = min(len(valid), probs.shape[1])
    probs_np = probs[0, :n_v].numpy()
    logits_np = logits[0, :n_v].numpy()

    # Sort by prob descending and look at top-K
    order = np.argsort(-probs_np)
    print(f'\nTop {args.top_k} model-preferred Phase 1b actions:')
    print(f'{"rank":>4} {"valid_idx":>9} {"prob":>8} {"is_p1a":>7} '
          f'{"recency":>8} {"raw_sz":>7} {"cached_sz":>10} '
          f'{"cached_in_expr_nm":>17} {"delta_norm":>11}')
    rank = 0
    p1b_in_top = []
    for ord_idx in order:
        rank += 1
        f = feats[ord_idx]
        if rank <= 20 or (rank <= args.top_k and not f['is_p1a']):
            print(f'{rank:>4} {f["idx"]:>9} {probs_np[ord_idx]:>8.4f} '
                  f'{str(f["is_p1a"]):>7} {f["anchor_recency"]:>8} '
                  f'{f["raw_size"]:>7} {f["cached_size"]:>10} '
                  f'{f["cached_in_expr_nm"]:>17} {f["delta_norm"]:>11}')
        if not f['is_p1a']:
            p1b_in_top.append(f)
        if rank >= args.top_k:
            break

    # Compute correlations across top-K Phase 1b actions
    if p1b_in_top:
        print(f'\nCorrelations on top-{args.top_k} Phase 1b actions ({len(p1b_in_top)} total):')
        prob_p1b = np.array([probs_np[f["idx"]] for f in p1b_in_top])
        for name in ('anchor_recency', 'raw_size', 'cached_size',
                     'cached_in_expr_nm', 'delta_norm'):
            vals = np.array([f[name] for f in p1b_in_top], dtype=float)
            if vals.std() > 0 and prob_p1b.std() > 0:
                corr = np.corrcoef(vals, prob_p1b)[0, 1]
                print(f'  pearson r(prob, {name:>20s}) = {corr:+.3f}')
            else:
                print(f'  {name}: stdev=0, skip')

    # Compare top-K Phase 1b features vs all Phase 1b features
    all_p1b_feats = [f for f in feats if not f['is_p1a']]
    print(f'\nAll Phase 1b feature distributions ({len(all_p1b_feats)} actions):')
    for name in ('anchor_recency', 'raw_size', 'cached_size',
                 'cached_in_expr_nm', 'delta_norm'):
        vals = np.array([f[name] for f in all_p1b_feats])
        print(f'  {name:>20s}: mean={vals.mean():.1f} std={vals.std():.1f} '
              f'min={vals.min()} max={vals.max()}')
    print(f'\nTop-{args.top_k} Phase 1b feature distributions '
          f'({len(p1b_in_top)} actions):')
    for name in ('anchor_recency', 'raw_size', 'cached_size',
                 'cached_in_expr_nm', 'delta_norm'):
        vals = np.array([f[name] for f in p1b_in_top])
        if len(vals):
            print(f'  {name:>20s}: mean={vals.mean():.1f} std={vals.std():.1f} '
                  f'min={vals.min()} max={vals.max()}')


if __name__ == '__main__':
    sys.exit(main())
