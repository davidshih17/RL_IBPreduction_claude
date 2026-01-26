#!/usr/bin/env python3
"""
Beam search reduction using classifier v4 with target conditioning.

The v4 model takes the target integral as explicit input, allowing it to
learn what action to take for reducing a specific target.

Usage:
    python -u scripts/eval/beam_search_classifier_v4_target.py --beam_width 10 --max_steps 100
"""

import sys
import argparse
from pathlib import Path
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import torch
from ibp_env import IBPEnvironment, set_prime, filter_top_sector, is_master, weight
from classifier_v4_target import IBPActionClassifierV4Target


State = namedtuple('State', ['expr', 'subs', 'score', 'path', 'n_non_masters'])


def get_sector_mask(integral):
    """Get 6-bit sector mask from integral indices."""
    return [1 if integral[i] > 0 else 0 for i in range(6)]


def prepare_model_input(expr, subs, valid_actions, target_sector, device):
    """Prepare model input tensors, filtering to target_sector (not hardcoded top sector)."""
    max_terms = 200
    max_subs = 50
    max_actions = 900

    # Filter expression to target sector
    sector_expr = filter_to_sector(expr, target_sector)

    # Expression tensors
    expr_integrals = torch.zeros(1, max_terms, 7, dtype=torch.long, device=device)
    expr_coeffs = torch.zeros(1, max_terms, dtype=torch.long, device=device)
    expr_mask = torch.zeros(1, max_terms, dtype=torch.bool, device=device)

    for j, (integral, coeff) in enumerate(list(sector_expr.items())[:max_terms]):
        expr_integrals[0, j] = torch.tensor(integral, device=device)
        expr_coeffs[0, j] = coeff
        expr_mask[0, j] = True

    # Substitution tensors
    sub_integrals = torch.zeros(1, max_subs, 7, dtype=torch.long, device=device)
    sub_mask = torch.zeros(1, max_subs, dtype=torch.bool, device=device)

    for j, integral in enumerate(list(subs.keys())[:max_subs]):
        sub_integrals[0, j] = torch.tensor(integral, device=device)
        sub_mask[0, j] = True

    # Action tensors
    action_ibp_ops = torch.zeros(1, max_actions, dtype=torch.long, device=device)
    action_deltas = torch.zeros(1, max_actions, 7, dtype=torch.long, device=device)
    action_mask = torch.zeros(1, max_actions, dtype=torch.bool, device=device)

    for j, (ibp_op, delta) in enumerate(valid_actions[:max_actions]):
        action_ibp_ops[0, j] = ibp_op
        action_deltas[0, j] = torch.tensor(delta, device=device)
        action_mask[0, j] = True

    return {
        'expr_integrals': expr_integrals,
        'expr_coeffs': expr_coeffs,
        'expr_mask': expr_mask,
        'sub_integrals': sub_integrals,
        'sub_mask': sub_mask,
        'action_ibp_ops': action_ibp_ops,
        'action_deltas': action_deltas,
        'action_mask': action_mask,
    }


def filter_to_sector(expr, sector):
    """Filter expr to integrals matching the given sector tuple."""
    return {k: v for k, v in expr.items() if get_sector_mask(k) == list(sector)}


def get_non_masters(expr, target_sector=None):
    """Get non-master integrals from specified sector (or top sector if None)."""
    if target_sector is None:
        filtered = filter_top_sector(expr)
    else:
        filtered = filter_to_sector(expr, target_sector)
    return {k: v for k, v in filtered.items() if not is_master(k)}


def max_weight(expr, target_sector=None):
    """Get the maximum weight among non-masters in expression."""
    nms = get_non_masters(expr, target_sector)
    if not nms:
        return (0, 0)
    return max((weight(k)[0], weight(k)[1]) for k in nms)


def beam_search(env, model, start_expr, beam_width=10, max_steps=100, device='cpu', verbose=True,
                target_sector=None, filter_mode='subsector'):
    """Beam search with top-k actions at each step.

    Args:
        target_sector: If provided, only reduce integrals in this sector.
                      If None, uses filter_top_sector (original behavior).
        filter_mode: 'subsector' (default) - strict filtering, no lateral sectors
                     'higher_only' - old filtering, allows lateral sectors
                     'none' - no sector filtering
    """

    initial_non_masters = get_non_masters(start_expr, target_sector)
    initial_state = State(
        expr=start_expr,
        subs={},
        score=0.0,
        path=[],
        n_non_masters=len(initial_non_masters)
    )

    beam = [initial_state]
    best_solution = None
    best_weight_ever = max_weight(start_expr, target_sector)
    initial_weight = best_weight_ever

    for step in range(max_steps):
        if not beam:
            break

        # Check for success and track best weight
        for state in beam:
            w = max_weight(state.expr, target_sector)
            if w < best_weight_ever:
                best_weight_ever = w
                if verbose:
                    print(f'Step {step}: NEW BEST WEIGHT {best_weight_ever} (started at {initial_weight})', flush=True)
                    nms = get_non_masters(state.expr, target_sector)
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
            non_masters = get_non_masters(state.expr, target_sector)
            if not non_masters:
                continue

            # Find max weight and get ALL non-masters with that weight (handle ties)
            max_w = max((weight(k)[0], weight(k)[1]) for k in non_masters.keys())
            tied_targets = [k for k in non_masters.keys() if (weight(k)[0], weight(k)[1]) == max_w]

            # Try actions on ALL tied targets (not just one)
            for target in tied_targets:
                valid_actions = env.get_valid_actions(target, state.subs, filter_mode=filter_mode)

                if not valid_actions:
                    continue

                # Get sector mask from target
                sector_mask = get_sector_mask(target)

                # Prepare model input - use our function that filters to target_sector, NOT env._prepare_model_input
                batch = prepare_model_input(state.expr, state.subs, valid_actions, target_sector, device)
                batch['sector_mask'] = torch.tensor([sector_mask], dtype=torch.long, device=device)
                batch['target_integral'] = torch.tensor([list(target)], dtype=torch.long, device=device)

                with torch.no_grad():
                    logits, probs = model(
                        batch['expr_integrals'],
                        batch['expr_coeffs'],
                        batch['expr_mask'],
                        batch['sub_integrals'],
                        batch['sub_mask'],
                        batch['action_ibp_ops'],
                        batch['action_deltas'],
                        batch['action_mask'],
                        batch['sector_mask'],
                        batch['target_integral']  # NEW: pass target to model
                    )

                # Get top-k actions (divide by number of tied targets to keep total candidates reasonable)
                n_valid = min(len(valid_actions), 1400)
                k_per_target = max(1, beam_width // len(tied_targets))
                top_k = min(k_per_target, n_valid)
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

                    new_non_masters = get_non_masters(new_expr, target_sector)
                    new_path = state.path + [(target, ibp_op, delta)]

                    # Score: cumulative log prob
                    new_score = state.score + torch.log(torch.tensor(action_prob + 1e-10)).item()

                    candidates.append(State(
                        expr=new_expr,
                        subs=new_subs,
                        score=new_score,
                        path=new_path,
                        n_non_masters=len(new_non_masters)
                    ))

        # Select top beam_width candidates
        candidates.sort(key=lambda s: (max_weight(s.expr, target_sector), s.n_non_masters, -s.score))
        beam = candidates[:beam_width]

        if verbose and step % 10 == 0:
            best = beam[0] if beam else None
            if best:
                w = max_weight(best.expr, target_sector)
                print(f'Step {step}: beam has {len(beam)} states, best has {best.n_non_masters} non-masters, max_weight={w}', flush=True)

    return best_solution, beam, best_weight_ever


def main():
    parser = argparse.ArgumentParser(description='Beam search reduction with classifier v4 target')
    parser.add_argument('--integral', type=str, default='2,0,2,0,1,1,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v4_target_sector37/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=10,
                        help='Number of top actions to explore at each step')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of reduction steps')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on')
    parser.add_argument('--filter_mode', type=str, default='subsector',
                        choices=['subsector', 'higher_only', 'none'],
                        help='Sector filtering mode: subsector (strict, no lateral), higher_only (old, allows lateral), none')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('Beam Search Reduction with Classifier v4 Target', flush=True)
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

    model = IBPActionClassifierV4Target(
        embed_dim=ckpt_args.get('embed_dim', 256),
        n_heads=ckpt_args.get('n_heads', 4),
        n_expr_layers=ckpt_args.get('n_expr_layers', 2),
        n_cross_layers=ckpt_args.get('n_cross_layers', 2),
        prime=ckpt_args.get('prime', args.prime)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(args.device)
    print(f'Loaded model from epoch {checkpoint["epoch"]} (val loss: {checkpoint["val_metrics"]["loss"]:.4f}, val acc: {checkpoint["val_metrics"]["top1_acc"]:.4f})', flush=True)

    # Parse starting integral
    indices = tuple(int(x) for x in args.integral.split(','))
    start_expr = {indices: 1}
    # Detect sector from input integral
    target_sector = tuple(get_sector_mask(indices))
    print(f'\nStarting integral: I{list(indices)}', flush=True)
    print(f'Initial weight: {weight(indices)}', flush=True)
    print(f'Sector mask: {list(target_sector)}', flush=True)

    # Run beam search
    print(f'\nRunning beam search (beam_width={args.beam_width}, max_steps={args.max_steps})...', flush=True)
    print('=' * 70, flush=True)

    solution, final_beam, best_weight = beam_search(
        env, model, start_expr,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        verbose=True,
        target_sector=target_sector,  # Pass the detected sector
        filter_mode=args.filter_mode
    )

    print('=' * 70, flush=True)
    print('\nResults:', flush=True)
    print(f'Best weight achieved: {best_weight}', flush=True)

    if solution:
        print(f'\nSUCCESS! Found complete reduction in {len(solution.path)} steps', flush=True)
        print('\nReduction path:', flush=True)
        for i, (target, ibp_op, delta) in enumerate(solution.path):
            print(f'  {i+1}. Eliminate I{list(target)} using IBP_{ibp_op}, delta={list(delta)}', flush=True)

        # Print the ENTIRE final expression
        print('\n' + '=' * 70, flush=True)
        print('FULL FINAL EXPRESSION (ALL INTEGRALS):', flush=True)
        print('=' * 70, flush=True)
        final_expr = solution.expr
        print(f'Total integrals in expression: {len(final_expr)}', flush=True)

        # Group by sector
        sectors = {}
        for integral, coeff in final_expr.items():
            sector = tuple(get_sector_mask(integral))
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append((integral, coeff))

        print(f'Number of distinct sectors: {len(sectors)}', flush=True)

        # Sort sectors by level (descending) then by value
        for sector in sorted(sectors.keys(), key=lambda s: (-sum(s), -sum(b << i for i, b in enumerate(s)))):
            level = sum(sector)
            integrals_in_sector = sectors[sector]
            print(f'\n--- Sector {list(sector)} (level {level}) - {len(integrals_in_sector)} integrals ---', flush=True)
            for integral, coeff in sorted(integrals_in_sector, key=lambda x: (-weight(x[0])[0], -weight(x[0])[1])):
                w = weight(integral)
                is_m = is_master(integral)
                print(f'  I{list(integral)} coeff={coeff} weight={w[:2]} master={is_m}', flush=True)
    else:
        print('\nNo complete reduction found.', flush=True)
        if final_beam:
            best = final_beam[0]
            print(f'Best state has {best.n_non_masters} non-masters remaining', flush=True)
            print(f'Path length: {len(best.path)} steps', flush=True)
            nms = get_non_masters(best.expr, target_sector)
            print(f'Remaining non-masters:', flush=True)
            for nm in sorted(nms.keys(), key=lambda x: (weight(x)[0], weight(x)[1]), reverse=True)[:10]:
                print(f'  I{list(nm)} weight={weight(nm)[:2]}', flush=True)


if __name__ == '__main__':
    main()
