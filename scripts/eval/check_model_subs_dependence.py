#!/usr/bin/env python3
"""Empirical test: does the model's forward pass depend on the `subs` input?

Loads a real dup_pair.pkl (two beam_search candidates with IDENTICAL
target_sector_expr and IDENTICAL resolved_subs, but DIFFERENT subs sequences —
they collided under the new dedup key). Builds the model input for each state
using prepare_batched_input_v5 EXACTLY as beam_search does, runs the model,
and reports whether the logits are bit-equal.

If logits differ → my claim that "model output depends on subs" is correct, and
the dedup-ON over-merging is explained.

If logits are bit-equal → I was wrong; something else is causing dedup-ON to
behave differently and I need to keep digging.
"""
import argparse
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
import numpy as np

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, weight, filter_subs_to_exact_sector,
    filter_resolved_subs_to_exact_sector,
)
from sailir.classifier import IBPActionClassifier
from beam_search import prepare_batched_input_v5
from beam_search_utils import get_sector_mask, get_non_masters, max_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True, type=str,
                        help='Path to dup_pair.pkl dumped by beam_search dedup debug')
    parser.add_argument('--topology', required=True, type=str)
    parser.add_argument('--model-checkpoint', required=True, type=str)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Setup
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    env = IBPEnvironment()

    # Load pair
    with open(args.pair, 'rb') as f:
        pair = pickle.load(f)
    state_a = pair['state_kept']
    state_b = pair['state_skipped']
    target_sector = pair['target_sector']

    print(f"=== State A (kept by dedup) ===")
    print(f"  |expr|={len(state_a.expr)} |subs|={len(state_a.subs)} "
          f"|resolved_subs|={len(state_a.resolved_subs)} score={state_a.score:.4f} "
          f"NM={state_a.n_non_masters} path_len={len(state_a.path)}")
    print(f"=== State B (skipped as duplicate) ===")
    print(f"  |expr|={len(state_b.expr)} |subs|={len(state_b.subs)} "
          f"|resolved_subs|={len(state_b.resolved_subs)} score={state_b.score:.4f} "
          f"NM={state_b.n_non_masters} path_len={len(state_b.path)}")
    print()

    # Sanity-check that the dedup key really matches
    expr_eq = state_a.expr == state_b.expr
    rs_eq = state_a.resolved_subs == state_b.resolved_subs
    subs_eq = state_a.subs == state_b.subs
    print(f"expr equal?         {expr_eq}")
    print(f"resolved_subs eq?   {rs_eq}")
    print(f"subs equal?         {subs_eq}")
    print()

    if not (expr_eq and rs_eq):
        print("ERROR: dedup key components don't match — this isn't a real duplicate pair.")
        sys.exit(1)

    # Load model
    checkpoint = torch.load(args.model_checkpoint, weights_only=False,
                            map_location=args.device)
    ckpt_args = checkpoint.get('args', {})
    model = IBPActionClassifier(
        embed_dim=ckpt_args.get('embed_dim', 256),
        n_heads=ckpt_args.get('n_heads', 4),
        n_expr_layers=ckpt_args.get('n_expr_layers', 2),
        n_cross_layers=ckpt_args.get('n_cross_layers', 2),
        n_subs_layers=ckpt_args.get('n_subs_layers', 2),
        prime=ckpt_args.get('prime', args.prime),
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(args.device)

    # Reproduce the beam_search flow that calls the model on each state.
    # First we need target + valid_actions for each. Since expr and
    # resolved_subs match, the target and valid_actions will match too.
    def get_target_and_actions(s):
        non_masters = get_non_masters(s.expr, target_sector)
        max_w = max((weight(k)[0], weight(k)[1]) for k in non_masters.keys())
        tied = [k for k in non_masters.keys() if (weight(k)[0], weight(k)[1]) == max_w]
        target = tied[0]
        filtered_subs = filter_subs_to_exact_sector(s.subs, target_sector)
        filtered_resolved = filter_resolved_subs_to_exact_sector(s.resolved_subs, target_sector)
        indirect_cache = env.compute_indirect_cache(filtered_subs, filtered_resolved)
        valid_actions = env.get_valid_actions_with_cache(
            target, indirect_cache, filtered_subs, filtered_resolved,
            filter_mode='subsector',
        )
        return target, valid_actions

    target_a, valid_a = get_target_and_actions(state_a)
    target_b, valid_b = get_target_and_actions(state_b)
    print(f"target A == target B? {target_a == target_b}")
    print(f"|valid_actions A| = {len(valid_a)}, |valid_actions B| = {len(valid_b)}")
    print(f"valid_actions equal? {valid_a == valid_b}")
    print()

    # Build model input for each state. Use the exact prepare_batched_input_v5
    # so we exercise the same code path beam_search does.
    batch_data_a = [(state_a.expr, state_a.subs, valid_a, target_sector, target_a)]
    batch_data_b = [(state_b.expr, state_b.subs, valid_b, target_sector, target_b)]
    batch_a, nva_a = prepare_batched_input_v5(batch_data_a, args.device)
    batch_b, nva_b = prepare_batched_input_v5(batch_data_b, args.device)

    # Are sub_* tensors actually different between A and B?
    sub_diff_keys = (batch_a['sub_keys'] != batch_b['sub_keys']).any().item()
    sub_diff_mask = (batch_a['sub_mask'] != batch_b['sub_mask']).any().item()
    sub_diff_ints = (batch_a['sub_repl_ints'] != batch_b['sub_repl_ints']).any().item()
    sub_diff_coeffs = (batch_a['sub_repl_coeffs'] != batch_b['sub_repl_coeffs']).any().item()
    sub_diff_rmask = (batch_a['sub_repl_mask'] != batch_b['sub_repl_mask']).any().item()
    expr_same = (batch_a['expr_integrals'] == batch_b['expr_integrals']).all().item() and \
                (batch_a['expr_coeffs'] == batch_b['expr_coeffs']).all().item() and \
                (batch_a['expr_mask'] == batch_b['expr_mask']).all().item()
    target_same = (batch_a['target_integral'] == batch_b['target_integral']).all().item()
    actions_same = (batch_a['action_ibp_ops'] == batch_b['action_ibp_ops']).all().item() and \
                   (batch_a['action_deltas'] == batch_b['action_deltas']).all().item() and \
                   (batch_a['action_mask'] == batch_b['action_mask']).all().item()
    print(f"Model-input tensors:")
    print(f"  expr_*       identical? {expr_same}")
    print(f"  target_*     identical? {target_same}")
    print(f"  action_*     identical? {actions_same}")
    print(f"  sub_keys     differ?    {sub_diff_keys}")
    print(f"  sub_mask     differ?    {sub_diff_mask}")
    print(f"  sub_repl_ints differ?   {sub_diff_ints}")
    print(f"  sub_repl_coeffs differ? {sub_diff_coeffs}")
    print(f"  sub_repl_mask differ?   {sub_diff_rmask}")
    print()

    # Forward the model on each. Compare logits.
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

    nv = nva_a[0]
    la = logits_a[0, :nv].cpu().numpy()
    lb = logits_b[0, :nv].cpu().numpy()
    pa = probs_a[0, :nv].cpu().numpy()
    pb = probs_b[0, :nv].cpu().numpy()
    diff = la - lb
    print(f"Logits over {nv} valid actions:")
    print(f"  max|logit_A - logit_B|   = {np.abs(diff).max():.6e}")
    print(f"  mean|logit_A - logit_B|  = {np.abs(diff).mean():.6e}")
    print(f"  bit-equal?               = {(la == lb).all()}")
    print(f"  argsort top-5 same?      = {(np.argsort(-la)[:5] == np.argsort(-lb)[:5]).all()}")
    top5_a = np.argsort(-la)[:5]
    top5_b = np.argsort(-lb)[:5]
    print(f"  top-5 idx A: {top5_a.tolist()}  probs: {pa[top5_a].round(4).tolist()}")
    print(f"  top-5 idx B: {top5_b.tolist()}  probs: {pb[top5_b].round(4).tolist()}")


if __name__ == '__main__':
    main()
