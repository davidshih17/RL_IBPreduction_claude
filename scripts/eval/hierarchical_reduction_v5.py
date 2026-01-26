#!/usr/bin/env python3
"""
Hierarchical reduction script for V5 model.

Reduces integral sector by sector from top to bottom using beam search.
Imports beam_search from beam_search_classifier_v5.py directly to avoid reimplementation.

Usage:
    python -u scripts/eval/hierarchical_reduction_v5.py \
        --integral 2,0,2,0,1,1,0 \
        --beam_width 20 \
        --max_steps 100 \
        --filter_mode subsector
"""

import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import torch
from ibp_env import IBPEnvironment, set_prime, is_master, weight, PRIME

# Import from beam_search_classifier_v5 - DO NOT REIMPLEMENT
from beam_search_classifier_v5 import (
    beam_search,
    prepare_model_input_v5,
    prepare_batched_input_v5,
)
from beam_search_classifier_v4_target import (
    State,
    get_sector_mask,
    filter_to_sector,
    get_non_masters,
    max_weight
)
from classifier_v5 import IBPActionClassifierV5


def sector_level(sector):
    """Return sum of 1s in sector (number of active propagators)."""
    return sum(sector)


def sector_to_int(sector):
    """Convert sector tuple to integer (for sorting)."""
    return sum(b << i for i, b in enumerate(sector))


def get_sectors_in_expr(expr):
    """Get all distinct sectors in expression."""
    sectors = set()
    for integral in expr.keys():
        sector = tuple(get_sector_mask(integral))
        sectors.add(sector)
    return sectors


def get_integrals_in_sector(expr, target_sector):
    """Get all integrals in a specific sector."""
    result = {}
    for integral, coeff in expr.items():
        sector = tuple(get_sector_mask(integral))
        if sector == target_sector:
            result[integral] = coeff
    return result


def get_non_masters_in_sector(expr, target_sector):
    """Get non-master integrals in a specific sector."""
    result = {}
    for integral, coeff in expr.items():
        sector = tuple(get_sector_mask(integral))
        if sector == target_sector and not is_master(integral):
            result[integral] = coeff
    return result


def print_expression_summary(expr, title="Expression"):
    """Print summary of expression grouped by sector."""
    if not expr:
        print(f"{title}: empty")
        return

    by_sector = defaultdict(list)
    for integral, coeff in expr.items():
        sector = tuple(get_sector_mask(integral))
        is_m = is_master(integral)
        w = weight(integral)
        by_sector[sector].append((integral, coeff, is_m, w))

    print(f"\n{title} ({len(expr)} integrals):")
    for sector in sorted(by_sector.keys(), key=lambda s: (-sector_level(s), -sector_to_int(s))):
        level = sector_level(sector)
        sint = sector_to_int(sector)
        integrals = by_sector[sector]
        n_masters = sum(1 for _, _, is_m, _ in integrals if is_m)
        n_non_masters = len(integrals) - n_masters
        print(f"  Sector {list(sector)} (level {level}, int {sint}): {len(integrals)} integrals ({n_masters} masters, {n_non_masters} non-masters)")
        for integral, coeff, is_m, w in sorted(integrals, key=lambda x: (-x[3][0], -x[3][1])):
            tag = "master=True" if is_m else "master=False"
            print(f"    I{list(integral)} coeff={coeff} weight={w[:2]} {tag}")


def hierarchical_reduction(env, model, start_expr, beam_width=20, max_steps=100,
                           max_iterations=1000, device='cpu', filter_mode='subsector', verbose=True):
    """
    Hierarchically reduce expression sector by sector.

    Strategy:
    1. Find highest-level sector with non-masters
    2. Reduce all non-masters in that sector using beam search
    3. Repeat until only masters remain
    """
    current_expr = dict(start_expr)
    current_subs = {}

    total_steps = 0
    iteration = 0
    sector_stats = []
    start_time = time.time()

    while iteration < max_iterations:
        iteration += 1

        # Find all sectors with non-masters
        sectors_with_non_masters = []
        for integral in current_expr.keys():
            if not is_master(integral):
                sector = tuple(get_sector_mask(integral))
                if sector not in sectors_with_non_masters:
                    sectors_with_non_masters.append(sector)

        if not sectors_with_non_masters:
            print(f"\nSUCCESS! All integrals are masters after {iteration-1} sector reductions.")
            break

        # Sort by level (descending) then by sector int (descending)
        sectors_with_non_masters.sort(key=lambda s: (-sector_level(s), -sector_to_int(s)))
        target_sector = sectors_with_non_masters[0]

        level = sector_level(target_sector)
        sint = sector_to_int(target_sector)

        print(f"\n{'='*70}")
        print(f"=== Iteration {iteration}: Sector {list(target_sector)} (level {level}, int {sint}) ===")
        print(f"{'='*70}")

        if verbose:
            print_expression_summary(current_expr, "Full expression")

        # Get non-masters in this sector
        non_masters = get_non_masters_in_sector(current_expr, target_sector)
        print(f"\nNon-masters in target sector: {len(non_masters)}")
        for nm, coeff in sorted(non_masters.items(), key=lambda x: (-weight(x[0])[0], -weight(x[0])[1])):
            print(f"  I{list(nm)} weight={weight(nm)[:2]}")

        # Run beam search to reduce this sector
        # We pass the FULL expression but tell beam search to focus on target_sector
        # Track sector 52 corner integral when reducing sector 52
        track_ints = None
        if target_sector == (0, 0, 1, 0, 1, 1):
            track_ints = [(0, 0, 1, 0, 1, 1, 0)]  # Track the sector 52 corner
            print(f"\n  >>> Tracking coefficient of I[0,0,1,0,1,1,0] <<<", flush=True)

        solution, final_beam, best_weight = beam_search(
            env, model, current_expr,
            beam_width=beam_width,
            max_steps=max_steps,
            device=device,
            verbose=verbose,
            target_sector=target_sector,
            filter_mode=filter_mode,
            track_integrals=track_ints
        )

        if solution:
            # Success - update expression
            current_expr = solution.expr
            current_subs = solution.subs
            steps_taken = len(solution.path)
            total_steps += steps_taken

            print(f"\nSector {list(target_sector)} reduced in {steps_taken} steps")
            sector_stats.append({
                'sector': target_sector,
                'level': level,
                'steps': steps_taken,
                'success': True
            })
        else:
            # Failed - try to continue with best state
            if final_beam:
                best_state = final_beam[0]
                current_expr = best_state.expr
                current_subs = best_state.subs
                steps_taken = len(best_state.path)
                total_steps += steps_taken

                remaining_non_masters = len(get_non_masters_in_sector(best_state.expr, target_sector))
                print(f"\nFAILED to fully reduce sector {list(target_sector)}")
                print(f"  Best state has {remaining_non_masters} non-masters remaining after {steps_taken} steps")
                print(f"  Best weight achieved: {best_weight}")

                sector_stats.append({
                    'sector': target_sector,
                    'level': level,
                    'steps': steps_taken,
                    'success': False,
                    'remaining': remaining_non_masters,
                    'best_weight': best_weight
                })

                # Check if we made any progress
                if remaining_non_masters >= len(non_masters):
                    print(f"  No progress made - stopping.")
                    break
            else:
                print(f"\nFATAL: No beam states remaining for sector {list(target_sector)}")
                break

    elapsed = time.time() - start_time

    # Print final summary
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL REDUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Total iterations: {iteration}")
    print(f"Total steps: {total_steps}")

    # Check final state
    remaining_non_masters = [i for i in current_expr.keys() if not is_master(i)]
    if remaining_non_masters:
        print(f"\nWARNING: {len(remaining_non_masters)} non-masters remaining:")
        for nm in sorted(remaining_non_masters, key=lambda x: (-weight(x)[0], -weight(x)[1]))[:10]:
            sector = tuple(get_sector_mask(nm))
            print(f"  I{list(nm)} sector={list(sector)} weight={weight(nm)[:2]}")
    else:
        print(f"\nSUCCESS! Expression reduced to masters only.")

    print(f"\nSector-by-sector statistics:")
    for stat in sector_stats:
        status = "OK" if stat['success'] else f"FAILED (remaining={stat.get('remaining', '?')})"
        print(f"  Sector {list(stat['sector'])} (level {stat['level']}): {stat['steps']} steps [{status}]")

    print_expression_summary(current_expr, "\nFinal expression")

    return current_expr, sector_stats


def main():
    parser = argparse.ArgumentParser(description='Hierarchical reduction with classifier v5')
    parser.add_argument('--integral', type=str, default='2,0,2,0,1,1,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v5/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=20,
                        help='Beam width for search')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per sector reduction')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Max number of sector iterations')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on')
    parser.add_argument('--filter_mode', type=str, default='subsector',
                        choices=['subsector', 'higher_only', 'none'],
                        help='Sector filtering mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    args = parser.parse_args()

    print('='*70)
    print('Hierarchical Reduction with Classifier v5')
    print('='*70)
    print(f'Config:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    print()

    # Set prime
    set_prime(args.prime)
    print(f'Using PRIME = {args.prime}')

    # Load environment
    print('Loading IBP environment...')
    env = IBPEnvironment()
    print(f'Loaded {len(env.ibp_t)} IBP templates, {len(env.li_t)} LI templates')

    # Load model
    print(f'\nLoading model from {args.checkpoint}...')
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
    print(f'Loaded model from epoch {checkpoint["epoch"]} (val acc: {checkpoint["val_metrics"]["top1_acc"]:.4f})')

    # Parse starting integral
    indices = tuple(int(x) for x in args.integral.split(','))
    start_expr = {indices: 1}

    print(f'\nStarting integral: I{list(indices)}')
    print(f'Initial weight: {weight(indices)}')
    print(f'Sector mask: {list(get_sector_mask(indices))}')

    # Run hierarchical reduction
    print(f'\n{"="*70}')
    print(f'Starting hierarchical reduction...')
    print(f'{"="*70}')

    final_expr, stats = hierarchical_reduction(
        env, model, start_expr,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        max_iterations=args.max_iterations,
        device=args.device,
        filter_mode=args.filter_mode,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
