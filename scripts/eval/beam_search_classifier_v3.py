#!/usr/bin/env python3
"""
Beam search reduction using classifier v3 model.

Uses the trained transformer model to score actions and explores top-k actions at each step.

Usage:
    python -u scripts/eval/beam_search_classifier_v3.py --beam_width 5 --max_steps 100

Run with logging:
    nohup python -u scripts/eval/beam_search_classifier_v3.py > logs/beam_search_v3.log 2>&1 &
"""

import sys
import argparse
from pathlib import Path
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import torch
from ibp_env import IBPEnvironment, set_prime, filter_top_sector, is_master, weight
from classifier_v3 import IBPActionClassifierV3


# State for beam search
State = namedtuple('State', ['expr', 'subs', 'score', 'path', 'n_non_masters'])


def get_non_masters(expr):
    """Get non-master integrals from top sector."""
    top = filter_top_sector(expr)
    return {k: v for k, v in top.items() if not is_master(k)}


def max_weight(expr):
    """Get the maximum weight among non-masters in expression."""
    nms = get_non_masters(expr)
    if not nms:
        return (0, 0)
    return max((weight(k)[0], weight(k)[1]) for k in nms)


def beam_search(env, model, start_expr, beam_width=5, max_steps=50, device='cpu', verbose=True):
    """Beam search with top-k actions at each step."""

    initial_non_masters = get_non_masters(start_expr)
    initial_state = State(
        expr=start_expr,
        subs={},
        score=0.0,
        path=[],
        n_non_masters=len(initial_non_masters)
    )

    beam = [initial_state]
    best_solution = None
    best_weight_ever = max_weight(start_expr)
    initial_weight = best_weight_ever

    for step in range(max_steps):
        if not beam:
            break

        # Check for success and track best weight
        for state in beam:
            w = max_weight(state.expr)
            if w < best_weight_ever:
                best_weight_ever = w
                if verbose:
                    print(f'Step {step}: NEW BEST WEIGHT {best_weight_ever} (started at {initial_weight})', flush=True)
                    nms = get_non_masters(state.expr)
                    print(f'  Non-masters ({len(nms)}):', flush=True)
                    for nm in sorted(nms.keys(), key=lambda x: (weight(x)[0], weight(x)[1]), reverse=True):
                        print(f'    I{list(nm)} weight={weight(nm)[:2]}', flush=True)
            if state.n_non_masters == 0:
                if best_solution is None or len(state.path) < len(best_solution.path):
                    best_solution = state
                    if verbose:
                        print(f'Found solution in {len(state.path)} steps!', flush=True)

        if best_solution:
            break

        # Expand all states in beam
        candidates = []

        for state in beam:
            non_masters = get_non_masters(state.expr)
            if not non_masters:
                continue

            # Pick target (highest weight)
            target = max(non_masters.keys(), key=lambda x: (weight(x)[0], weight(x)[1]))
            valid_actions = env.get_valid_actions(target, state.subs)

            if not valid_actions:
                continue

            # Get model scores
            batch = env._prepare_model_input(state.expr, state.subs, valid_actions, device)
            with torch.no_grad():
                logits, probs = model(
                    batch['expr_integrals'],
                    batch['expr_coeffs'],
                    batch['expr_mask'],
                    batch['sub_integrals'],
                    batch['sub_mask'],
                    batch['action_ibp_ops'],
                    batch['action_deltas'],
                    batch['action_mask']
                )

            # Get top-k actions
            n_valid = min(len(valid_actions), 900)
            top_k = min(beam_width, n_valid)
            top_indices = torch.argsort(logits[0, :n_valid], descending=True)[:top_k].tolist()

            for idx in top_indices:
                ibp_op, delta = valid_actions[idx]
                action_prob = probs[0, idx].item()

                # Apply action
                new_expr, new_subs, success = env.apply_action(
                    dict(state.expr), dict(state.subs), target, ibp_op, delta
                )

                if not success:
                    continue

                new_non_masters = get_non_masters(new_expr)
                new_path = state.path + [(target, ibp_op, delta)]

                # Score: cumulative log prob (prefer higher prob paths)
                new_score = state.score + torch.log(torch.tensor(action_prob + 1e-10)).item()

                candidates.append(State(
                    expr=new_expr,
                    subs=new_subs,
                    score=new_score,
                    path=new_path,
                    n_non_masters=len(new_non_masters)
                ))

        # Select top beam_width candidates (prefer lower max weight, then fewer non-masters, then higher score)
        candidates.sort(key=lambda s: (max_weight(s.expr), s.n_non_masters, -s.score))
        beam = candidates[:beam_width]

        if verbose and step % 10 == 0:
            best = beam[0] if beam else None
            if best:
                w = max_weight(best.expr)
                print(f'Step {step}: beam has {len(beam)} states, best has {best.n_non_masters} non-masters, max_weight={w}', flush=True)

    return best_solution, beam, best_weight_ever


def main():
    parser = argparse.ArgumentParser(description='Beam search reduction with classifier v3')
    parser.add_argument('--integral', type=str, default='2,0,2,0,1,1,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v3_filtered/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Number of top actions to explore at each step')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of reduction steps')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('Beam Search Reduction with Classifier v3', flush=True)
    print('=' * 70, flush=True)
    print(f'Config:', flush=True)
    for k, v in vars(args).items():
        print(f'  {k}: {v}', flush=True)
    print(flush=True)

    # Set prime
    set_prime(args.prime)
    print(f'Using PRIME = {args.prime}', flush=True)

    # Load environment
    print('Loading IBP environment...', flush=True)
    env = IBPEnvironment()
    print(f'Loaded {len(env.ibp_t)} IBP templates, {len(env.li_t)} LI templates', flush=True)

    # Load model
    print(f'\nLoading model from {args.checkpoint}...', flush=True)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    ckpt_args = checkpoint.get('args', {})

    model = IBPActionClassifierV3(
        embed_dim=ckpt_args.get('embed_dim', 256),
        n_heads=ckpt_args.get('n_heads', 4),
        n_expr_layers=ckpt_args.get('n_expr_layers', 2),
        n_cross_layers=ckpt_args.get('n_cross_layers', 2),
        prime=ckpt_args.get('prime', args.prime)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(args.device)
    print(f'Loaded model from epoch {checkpoint["epoch"]} (val acc: {checkpoint["val_metrics"]["top1_acc"]:.4f})', flush=True)

    # Parse starting integral
    indices = tuple(int(x) for x in args.integral.split(','))
    start_expr = {indices: 1}
    print(f'\nStarting integral: I{list(indices)}', flush=True)
    print(f'Initial weight: {weight(indices)}', flush=True)

    # Run beam search
    print(f'\nRunning beam search (beam_width={args.beam_width}, max_steps={args.max_steps})...', flush=True)
    print('=' * 70, flush=True)

    solution, final_beam, best_weight = beam_search(
        env, model, start_expr,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        verbose=True
    )

    print('=' * 70, flush=True)
    print('\nResults:', flush=True)
    print(f'Best weight achieved: {best_weight}', flush=True)

    if solution:
        print(f'\nSUCCESS! Found complete reduction in {len(solution.path)} steps', flush=True)
        print('\nReduction path:', flush=True)
        for i, (target, ibp_op, delta) in enumerate(solution.path):
            print(f'  {i+1}. Eliminate I{list(target)} using IBP_{ibp_op}, delta={list(delta)}', flush=True)
    else:
        print('\nNo complete reduction found.', flush=True)
        if final_beam:
            best = final_beam[0]
            print(f'Best state has {best.n_non_masters} non-masters remaining', flush=True)
            print(f'Path length: {len(best.path)} steps', flush=True)
            nms = get_non_masters(best.expr)
            print(f'Remaining non-masters:', flush=True)
            for nm in sorted(nms.keys(), key=lambda x: (weight(x)[0], weight(x)[1]), reverse=True)[:10]:
                print(f'  I{list(nm)} weight={weight(nm)[:2]}', flush=True)


if __name__ == '__main__':
    main()
