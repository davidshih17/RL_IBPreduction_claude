#!/usr/bin/env python3
"""
Beam search reduction using classifier v5 with full substitution encoding.

The v5 model encodes substitutions fully:
- The integral being substituted (key)
- ALL replacement terms with their coefficients

Optimizations:
- Batched model inference using numpy arrays
- Cached get_raw_equation results

Usage:
    python -u scripts/eval/beam_search_classifier_v5.py --beam_width 10 --max_steps 100
"""

import sys
import argparse
from pathlib import Path
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import numpy as np
import torch
from ibp_env import IBPEnvironment, set_prime, filter_top_sector, is_master, weight
from classifier_v5 import IBPActionClassifierV5

# Reuse core functions from v4
from beam_search_classifier_v4_target import (
    State,
    get_sector_mask,
    filter_to_sector,
    get_non_masters,
    max_weight
)


MAX_REPLACEMENT_TERMS = 20  # Must match training


def prepare_model_input_v5(expr, subs, valid_actions, target_sector, device):
    """
    Prepare model input tensors for V5 model.

    Key difference from v4: substitutions include full replacement terms.
    subs format: {target_integral: {repl_integral: coeff, ...}, ...}
    """
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

    # Substitution tensors for V5 - full structure
    sub_keys = torch.zeros(1, max_subs, 7, dtype=torch.long, device=device)
    sub_repl_ints = torch.zeros(1, max_subs, MAX_REPLACEMENT_TERMS, 7, dtype=torch.long, device=device)
    sub_repl_coeffs = torch.zeros(1, max_subs, MAX_REPLACEMENT_TERMS, dtype=torch.long, device=device)
    sub_repl_mask = torch.zeros(1, max_subs, MAX_REPLACEMENT_TERMS, dtype=torch.bool, device=device)
    sub_mask = torch.zeros(1, max_subs, dtype=torch.bool, device=device)

    for j, (key_integral, replacement) in enumerate(list(subs.items())[:max_subs]):
        sub_mask[0, j] = True
        sub_keys[0, j] = torch.tensor(key_integral, device=device)

        # replacement is {integral: coeff, ...}
        for r, (repl_int, coeff) in enumerate(list(replacement.items())[:MAX_REPLACEMENT_TERMS]):
            sub_repl_ints[0, j, r] = torch.tensor(repl_int, device=device)
            sub_repl_coeffs[0, j, r] = coeff
            sub_repl_mask[0, j, r] = True

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
        'sub_keys': sub_keys,
        'sub_repl_ints': sub_repl_ints,
        'sub_repl_coeffs': sub_repl_coeffs,
        'sub_repl_mask': sub_repl_mask,
        'sub_mask': sub_mask,
        'action_ibp_ops': action_ibp_ops,
        'action_deltas': action_deltas,
        'action_mask': action_mask,
    }


def prepare_batched_input_v5(batch_data, device):
    """
    Prepare batched model input for V5 model using numpy for speed.

    Uses numpy arrays first, then converts to torch tensors once at the end
    to avoid thousands of torch.tensor() calls in Python loops.
    """
    batch_size = len(batch_data)

    # Find max sizes across batch
    max_terms = 200
    max_subs = 50
    max_actions = max(len(d[2]) for d in batch_data)  # valid_actions
    max_actions = min(max_actions, 900)

    # Allocate numpy arrays (much faster than torch tensors for item assignment)
    expr_integrals_np = np.zeros((batch_size, max_terms, 7), dtype=np.int64)
    expr_coeffs_np = np.zeros((batch_size, max_terms), dtype=np.int64)
    expr_mask_np = np.zeros((batch_size, max_terms), dtype=np.bool_)

    sub_keys_np = np.zeros((batch_size, max_subs, 7), dtype=np.int64)
    sub_repl_ints_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS, 7), dtype=np.int64)
    sub_repl_coeffs_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.int64)
    sub_repl_mask_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.bool_)
    sub_mask_np = np.zeros((batch_size, max_subs), dtype=np.bool_)

    action_ibp_ops_np = np.zeros((batch_size, max_actions), dtype=np.int64)
    action_deltas_np = np.zeros((batch_size, max_actions, 7), dtype=np.int64)
    action_mask_np = np.zeros((batch_size, max_actions), dtype=np.bool_)

    sector_masks_np = np.zeros((batch_size, 6), dtype=np.int64)
    target_integrals_np = np.zeros((batch_size, 7), dtype=np.int64)

    # Track number of valid actions per sample
    n_valid_actions = []

    for i, (expr, subs, valid_actions, target_sector, target) in enumerate(batch_data):
        # Filter expression to target sector
        sector_expr = filter_to_sector(expr, target_sector)

        # Expression - use numpy array assignment
        expr_items = list(sector_expr.items())[:max_terms]
        n_expr = len(expr_items)
        if n_expr > 0:
            expr_integrals_np[i, :n_expr] = [integral for integral, _ in expr_items]
            expr_coeffs_np[i, :n_expr] = [coeff for _, coeff in expr_items]
            expr_mask_np[i, :n_expr] = True

        # Substitutions (full structure for V5)
        subs_items = list(subs.items())[:max_subs]
        for j, (key_integral, replacement) in enumerate(subs_items):
            sub_mask_np[i, j] = True
            sub_keys_np[i, j] = key_integral
            repl_items = list(replacement.items())[:MAX_REPLACEMENT_TERMS]
            n_repl = len(repl_items)
            if n_repl > 0:
                sub_repl_ints_np[i, j, :n_repl] = [ri for ri, _ in repl_items]
                sub_repl_coeffs_np[i, j, :n_repl] = [c for _, c in repl_items]
                sub_repl_mask_np[i, j, :n_repl] = True

        # Actions - use numpy array assignment
        n_actions = min(len(valid_actions), max_actions)
        n_valid_actions.append(n_actions)
        if n_actions > 0:
            action_ibp_ops_np[i, :n_actions] = [op for op, _ in valid_actions[:n_actions]]
            action_deltas_np[i, :n_actions] = [delta for _, delta in valid_actions[:n_actions]]
            action_mask_np[i, :n_actions] = True

        # Sector and target
        sector_masks_np[i] = get_sector_mask(target)
        target_integrals_np[i] = target

    # Convert to torch tensors once (single transfer to GPU)
    return {
        'expr_integrals': torch.from_numpy(expr_integrals_np).to(device),
        'expr_coeffs': torch.from_numpy(expr_coeffs_np).to(device),
        'expr_mask': torch.from_numpy(expr_mask_np).to(device),
        'sub_keys': torch.from_numpy(sub_keys_np).to(device),
        'sub_repl_ints': torch.from_numpy(sub_repl_ints_np).to(device),
        'sub_repl_coeffs': torch.from_numpy(sub_repl_coeffs_np).to(device),
        'sub_repl_mask': torch.from_numpy(sub_repl_mask_np).to(device),
        'sub_mask': torch.from_numpy(sub_mask_np).to(device),
        'action_ibp_ops': torch.from_numpy(action_ibp_ops_np).to(device),
        'action_deltas': torch.from_numpy(action_deltas_np).to(device),
        'action_mask': torch.from_numpy(action_mask_np).to(device),
        'sector_mask': torch.from_numpy(sector_masks_np).to(device),
        'target_integral': torch.from_numpy(target_integrals_np).to(device),
    }, n_valid_actions


def beam_search(env, model, start_expr, beam_width=10, max_steps=100, device='cpu', verbose=True,
                target_sector=None, filter_mode='subsector', track_integrals=None):
    """Beam search with top-k actions at each step.

    OPTIMIZED:
    - Batched model inference using numpy arrays
    - Cached get_raw_equation results

    Args:
        target_sector: If provided, only reduce integrals in this sector.
                      If None, uses filter_top_sector (original behavior).
        filter_mode: 'subsector' (default) - strict filtering, no lateral sectors
                     'higher_only' - old filtering, allows lateral sectors
                     'none' - no sector filtering
        track_integrals: Optional list of integrals to track coefficients for.
    """
    import time

    # Initialize tracking
    if track_integrals:
        tracked_coeffs = {i: start_expr.get(i, 0) for i in track_integrals}
        print(f"  [TRACK] Initial coefficients:", flush=True)
        for i, c in tracked_coeffs.items():
            print(f"    I{list(i)} = {c}", flush=True)

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
        step_start = time.time()
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

        # PHASE 1: Collect all (state, target) pairs and get valid actions (cached)
        t1 = time.time()

        # First pass: collect all (state_idx, state, target) tuples
        tasks = []  # List of (state_idx, state, target, n_tied_targets)
        for state_idx, state in enumerate(beam):
            non_masters = get_non_masters(state.expr, target_sector)
            if not non_masters:
                continue

            # Find max weight and get ALL non-masters with that weight (handle ties)
            max_w = max((weight(k)[0], weight(k)[1]) for k in non_masters.keys())
            tied_targets = [k for k in non_masters.keys() if (weight(k)[0], weight(k)[1]) == max_w]

            for target in tied_targets:
                tasks.append((state_idx, state, target, len(tied_targets)))

        # Get valid actions - use cached version for better performance
        valid_actions_list = [
            env.get_valid_actions_cached(t[2], t[1].subs, filter_mode=filter_mode)
            for t in tasks
        ]

        # Build batch_data and batch_meta from results
        batch_data = []  # List of (expr, subs, valid_actions, target_sector, target)
        batch_meta = []  # List of (state_idx, target, valid_actions, n_tied_targets)

        for (state_idx, state, target, n_tied_targets), valid_actions in zip(tasks, valid_actions_list):
            if not valid_actions:
                continue
            batch_data.append((state.expr, state.subs, valid_actions, target_sector, target))
            batch_meta.append((state_idx, target, valid_actions, n_tied_targets))

        if not batch_data:
            break
        t1_elapsed = time.time() - t1

        # PHASE 2: Batched model inference
        t2 = time.time()
        batch, n_valid_actions = prepare_batched_input_v5(batch_data, device)

        with torch.no_grad():
            logits, probs = model(
                batch['expr_integrals'],
                batch['expr_coeffs'],
                batch['expr_mask'],
                batch['sub_keys'],
                batch['sub_repl_ints'],
                batch['sub_repl_coeffs'],
                batch['sub_repl_mask'],
                batch['sub_mask'],
                batch['action_ibp_ops'],
                batch['action_deltas'],
                batch['action_mask'],
                batch['sector_mask'],
                batch['target_integral']
            )
        t2_elapsed = time.time() - t2

        # PHASE 3: Extract top-k actions and create candidates
        t3 = time.time()
        candidates = []

        for i, (state_idx, target, valid_actions, n_tied_targets) in enumerate(batch_meta):
            state = beam[state_idx]
            n_valid = n_valid_actions[i]

            # Get top-k actions
            k_per_target = max(1, beam_width // n_tied_targets)
            top_k = min(k_per_target, n_valid)
            top_indices = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()

            for idx in top_indices:
                ibp_op, delta = valid_actions[idx]
                action_prob = probs[i, idx].item()

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
        t3_elapsed = time.time() - t3
        step_elapsed = time.time() - step_start

        # Track specific integrals
        if track_integrals and beam:
            best = beam[0]
            for i in track_integrals:
                new_coeff = best.expr.get(i, 0)
                old_coeff = tracked_coeffs[i]
                if new_coeff != old_coeff:
                    # Find what action was taken
                    if best.path:
                        last_target, last_ibp_op, last_delta = best.path[-1]
                        action_str = f"elim I{list(last_target)} via IBP_{last_ibp_op}"
                    else:
                        action_str = "initial"
                    print(f"  [TRACK] Step {step}: I{list(i)} coeff {old_coeff} -> {new_coeff} ({action_str})", flush=True)
                    tracked_coeffs[i] = new_coeff

        if verbose:
            print(f'Step {step}: P1(get_valid)={t1_elapsed:.2f}s P2(model)={t2_elapsed:.2f}s P3(apply)={t3_elapsed:.2f}s total={step_elapsed:.2f}s batch={len(batch_data)} cands={len(candidates)}', flush=True)
            if step % 10 == 0:
                best = beam[0] if beam else None
                if best:
                    w = max_weight(best.expr, target_sector)
                    print(f'  -> beam has {len(beam)} states, best has {best.n_non_masters} non-masters, max_weight={w}', flush=True)

    return best_solution, beam, best_weight_ever


def main():
    parser = argparse.ArgumentParser(description='Beam search reduction with classifier v5')
    parser.add_argument('--integral', type=str, default='2,0,1,0,2,0,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v5/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=10,
                        help='Number of top actions to explore at each step')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of reduction steps')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on')
    parser.add_argument('--filter_mode', type=str, default='higher_only',
                        choices=['subsector', 'higher_only', 'none'],
                        help='Sector filtering mode: subsector (strict, no lateral), higher_only (matches training), none')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('Beam Search Reduction with Classifier v5', flush=True)
    print('  - Target as input', flush=True)
    print('  - Full substitution encoding', flush=True)
    print('  - Cached get_raw_equation', flush=True)
    print('  - Batched model inference', flush=True)
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

    model = IBPActionClassifierV5(
        embed_dim=ckpt_args.get('embed_dim', 256),
        n_heads=ckpt_args.get('n_heads', 4),
        n_expr_layers=ckpt_args.get('n_expr_layers', 2),
        n_cross_layers=ckpt_args.get('n_cross_layers', 2),
        n_subs_layers=ckpt_args.get('n_subs_layers', 2),
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
        target_sector=target_sector,
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
