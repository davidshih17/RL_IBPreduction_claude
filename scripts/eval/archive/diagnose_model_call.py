#!/usr/bin/env python
"""Load step-26 ckpts of A and B, reconstruct the divergent survivor's
parent on both sides, call the model, compare logits/probs for the
chosen action.
"""
import argparse
import pickle
import sys
import torch
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
from beam_search_full import prepare_batched_input_v5
from beam_search_utils import get_non_masters
from sailir.classifier import IBPActionClassifier


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def build_iraws_with_cached(flat, rs, env):
    """Rebuild iraws as list of (sub_int, op, shift, raw, cached, ubm)."""
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
    p.add_argument('--model', required=True)
    p.add_argument('--target', required=True,
                   help='target as comma-sep, e.g. "0,1,2,0,1,1,2,1,-4,0,0"')
    p.add_argument('--action-op', type=int, required=True)
    p.add_argument('--action-delta', required=True)
    p.add_argument('--path-prefix-step', type=int, default=26)
    args = p.parse_args()

    target = tuple(int(x) for x in args.target.split(','))
    chosen_delta = tuple(int(x) for x in args.action_delta.split(','))

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    torch.set_num_threads(1)

    ca = load(args.ckpt_a)
    cb = load(args.ckpt_b)
    target_sector = tuple(ca['target_sector'])

    # Find the survivor whose expr contains target and whose path matches between A and B.
    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}
    common = sorted(set(by_a) & set(by_b))
    matched_path = None
    for p_ in common:
        sa = by_a[p_]
        sb = by_b[p_]
        if target in sa['expr']:
            matched_path = p_
            break
    if matched_path is None:
        print('No common survivor has target in expr.')
        return 1

    sa = by_a[matched_path]
    sb = by_b[matched_path]
    print(f'Selected survivor (matched path): '
          f'max_w={sa["max_w"]} nm={sa["n_non_masters"]} '
          f'score_A={sa["score"]!r} score_B={sb["score"]!r}')

    # Build iraws + run production enumerate for both.
    iraws_a = build_iraws_with_cached(sa['aux_flat'], sa['resolved_subs'], env)
    iraws_b = build_iraws_with_cached(sb['aux_flat'], sb['resolved_subs'], env)
    print(f'|iraws_A|={len(iraws_a)}  |iraws_B|={len(iraws_b)}')

    valid_a = enumerate_valid_actions_with_indirect_cache(
        target, iraws_a, sa['subs'], sa['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    valid_b = enumerate_valid_actions_with_indirect_cache(
        target, iraws_b, sb['subs'], sb['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    print(f'|valid_A|={len(valid_a)}  |valid_B|={len(valid_b)}')
    print(f'valid sets equal: {set(valid_a) == set(valid_b)}')
    print(f'valid lists equal (order too): {valid_a == valid_b}')

    chosen = (args.action_op, chosen_delta)
    try:
        idx_a = valid_a.index(chosen)
        idx_b = valid_b.index(chosen)
    except ValueError as e:
        print(f'Chosen action not in valid lists: {e}')
        return 1
    print(f'Chosen action idx: A={idx_a}  B={idx_b}')

    # Call model on each (state, target) ALONE (batch size 1).
    bd_a = [(sa['expr'], sa['subs'], valid_a, target_sector, target)]
    bd_b = [(sb['expr'], sb['subs'], valid_b, target_sector, target)]
    batch_a, _ = prepare_batched_input_v5(bd_a, 'cpu')
    batch_b, _ = prepare_batched_input_v5(bd_b, 'cpu')

    with torch.no_grad():
        logits_a, probs_a = model(
            batch_a['expr_integrals'], batch_a['expr_coeffs'], batch_a['expr_mask'],
            batch_a['sub_keys'], batch_a['sub_repl_ints'], batch_a['sub_repl_coeffs'],
            batch_a['sub_repl_mask'], batch_a['sub_mask'],
            batch_a['action_ibp_ops'], batch_a['action_deltas'], batch_a['action_mask'],
            batch_a['sector_mask'], batch_a['target_integral'],
        )
        logits_b, probs_b = model(
            batch_b['expr_integrals'], batch_b['expr_coeffs'], batch_b['expr_mask'],
            batch_b['sub_keys'], batch_b['sub_repl_ints'], batch_b['sub_repl_coeffs'],
            batch_b['sub_repl_mask'], batch_b['sub_mask'],
            batch_b['action_ibp_ops'], batch_b['action_deltas'], batch_b['action_mask'],
            batch_b['sector_mask'], batch_b['target_integral'],
        )

    n_v = len(valid_a)
    print(f'\nLogits at chosen idx: A={logits_a[0, idx_a].item()!r}  '
          f'B={logits_b[0, idx_b].item()!r}')
    print(f'Probs at chosen idx:  A={probs_a[0, idx_a].item()!r}  '
          f'B={probs_b[0, idx_b].item()!r}')
    print(f'log(prob) at chosen:  A={torch.log(probs_a[0, idx_a]).item()!r}  '
          f'B={torch.log(probs_b[0, idx_b]).item()!r}')
    delta = torch.log(probs_a[0, idx_a]).item() - torch.log(probs_b[0, idx_b]).item()
    print(f'log-prob delta: {delta:+.6e}')

    # Verify logits-set equality (sorted)
    la = sorted(logits_a[0, :n_v].tolist())
    lb = sorted(logits_b[0, :n_v].tolist())
    print(f'\nSorted logits match exactly: {la == lb}')
    if la != lb:
        # Find first mismatch in sorted lists
        for i in range(min(len(la), len(lb))):
            if la[i] != lb[i]:
                print(f'  First sorted mismatch at rank {i}: '
                      f'A={la[i]!r}  B={lb[i]!r}  delta={la[i]-lb[i]:+.6e}')
                # Show neighbors
                for j in range(max(0, i-2), min(len(la), i+3)):
                    print(f'    [{j}] A={la[j]!r}  B={lb[j]!r}')
                break

    # Also check sums (the softmax denominator)
    import math
    max_l = max(la[-1], lb[-1])
    sum_a = sum(math.exp(l - max_l) for l in la)
    sum_b = sum(math.exp(l - max_l) for l in lb)
    print(f'\nSoftmax denominators (after shifting by max):')
    print(f'  sum_A={sum_a!r}  sum_B={sum_b!r}  delta={sum_a-sum_b:+.6e}')

    # Compare action embeddings (small sanity check) - if the action emb
    # for the chosen action is bit-identical, then the per-action logit
    # is bit-identical (as we've seen).
    return 0


if __name__ == '__main__':
    sys.exit(main())
