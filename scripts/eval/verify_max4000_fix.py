#!/usr/bin/env python
"""Verify the max_actions=4000 fix via diff checkpoint replay.

Take v4-rescue and baseline thick checkpoints at step K (where they share
identical state). For each (survivor, target):
  1. Enumerate v4's valid and baseline's valid using production code.
  2. Run model with max_actions=4000 (no truncation since valid <= 4000 fits).
  3. For each action present in both valid lists, compare its model
     probability between v4-side and base-side.

If model is permutation-equivariant AND truncation is the only issue, every
shared action should get IDENTICAL probability in both runs \u2192 no score drift
\u2192 v4-rescue would bit-match baseline if rerun with max_actions=4000.
"""
import argparse
import pickle
import sys
import torch

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


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def build_iraws(flat, rs, env):
    n_idx = ibp_env.N_INDICES
    out = []
    for row in flat.iraws_meta:
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
    p.add_argument('ckpt_v4')
    p.add_argument('ckpt_base')
    p.add_argument('topology')
    p.add_argument('--model', required=True)
    p.add_argument('--max-survivors', type=int, default=40)
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

    ca = load(args.ckpt_v4)
    cb = load(args.ckpt_base)
    target_sector = tuple(ca['target_sector'])
    print(f'v4 step={ca["step"]} base step={cb["step"]}')

    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}
    common = sorted(set(by_a) & set(by_b))[:args.max_survivors]

    # Confirm state identicality across common survivors
    state_match = sum(
        1 for p_ in common
        if by_a[p_]['expr'] == by_b[p_]['expr']
        and by_a[p_]['subs'] == by_b[p_]['subs']
        and by_a[p_]['resolved_subs'] == by_b[p_]['resolved_subs']
    )
    print(f'Survivors with identical (expr, subs, RS): {state_match}/{len(common)}')

    n_tasks = 0
    n_action_prob_match = 0
    n_action_prob_close = 0  # within 1e-6
    n_action_prob_diff = 0
    max_prob_diff = 0.0
    n_actions_compared = 0

    for s_idx, p_ in enumerate(common):
        sa = by_a[p_]
        sb = by_b[p_]
        nm = get_non_masters(sa['expr'], target_sector)
        if not nm:
            continue
        mw = tuple(sa['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        if not tied:
            continue

        ia = build_iraws(sa['aux_flat'], sa['resolved_subs'], env)
        ib = build_iraws(sb['aux_flat'], sb['resolved_subs'], env)

        for target in tied:
            n_tasks += 1
            va = enumerate_valid_actions_with_indirect_cache(
                target, ia, sa['subs'], sa['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            vb = enumerate_valid_actions_with_indirect_cache(
                target, ib, sb['subs'], sb['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            # Ensure set-equal (sanity)
            assert set(va) == set(vb), 'valid sets differ'

            # Run model with max_actions=4000 on both
            bd_a = [(sa['expr'], sa['subs'], va, target_sector, target)]
            bd_b = [(sb['expr'], sb['subs'], vb, target_sector, target)]
            ba, _ = prepare_batched_input_v5(bd_a, 'cpu')
            bb_, _ = prepare_batched_input_v5(bd_b, 'cpu')
            with torch.no_grad():
                _, pa = model(
                    ba['expr_integrals'], ba['expr_coeffs'], ba['expr_mask'],
                    ba['sub_keys'], ba['sub_repl_ints'], ba['sub_repl_coeffs'],
                    ba['sub_repl_mask'], ba['sub_mask'],
                    ba['action_ibp_ops'], ba['action_deltas'], ba['action_mask'],
                    ba['sector_mask'], ba['target_integral'],
                )
                _, pb = model(
                    bb_['expr_integrals'], bb_['expr_coeffs'], bb_['expr_mask'],
                    bb_['sub_keys'], bb_['sub_repl_ints'], bb_['sub_repl_coeffs'],
                    bb_['sub_repl_mask'], bb_['sub_mask'],
                    bb_['action_ibp_ops'], bb_['action_deltas'], bb_['action_mask'],
                    bb_['sector_mask'], bb_['target_integral'],
                )

            # Model only scores first min(|valid|, 4000) actions; restrict
            # to actions that fall in BOTH runs' first 4000 windows.
            MAX = pa.shape[1]
            va_seen = set(va[:MAX])
            vb_seen = set(vb[:MAX])
            both_seen = va_seen & vb_seen
            # Map each shared-and-seen action to its index in A and B.
            pos_a = {act: i for i, act in enumerate(va[:MAX])}
            pos_b = {act: i for i, act in enumerate(vb[:MAX])}
            for act in both_seen:
                ia_idx = pos_a[act]
                ib_idx = pos_b[act]
                prob_a = pa[0, ia_idx].item()
                prob_b = pb[0, ib_idx].item()
                n_actions_compared += 1
                diff = abs(prob_a - prob_b)
                max_prob_diff = max(max_prob_diff, diff)
                if prob_a == prob_b:
                    n_action_prob_match += 1
                elif diff < 1e-6:
                    n_action_prob_close += 1
                else:
                    n_action_prob_diff += 1

    print(f'\nTasks tested: {n_tasks}')
    print(f'Actions compared (per-action prob across both runs): {n_actions_compared}')
    print(f'  bit-identical prob:              {n_action_prob_match}')
    print(f'  diff < 1e-6 (fp noise):          {n_action_prob_close}')
    print(f'  diff >= 1e-6:                    {n_action_prob_diff}')
    print(f'  max prob diff observed:          {max_prob_diff:.3e}')

    if n_action_prob_diff == 0 and n_action_prob_close == 0:
        print('\n*** PERFECT: all action probabilities BIT-IDENTICAL between v4 '
              'and baseline at max_actions=4000. ***')
        print('Means: v4-rescue rerun with max_actions=4000 would give '
              'bit-identical scores \u2192 bit-identical beam \u2192 no divergence.')
    elif n_action_prob_diff == 0:
        print('\nAll diffs < 1e-6 (numerical noise). Probabilities effectively '
              'identical; cumulative score drift would be tiny.')
    else:
        print(f'\n{n_action_prob_diff} action probs still differ \u2014 max_actions=4000 '
              f'fix may not fully eliminate divergence.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
