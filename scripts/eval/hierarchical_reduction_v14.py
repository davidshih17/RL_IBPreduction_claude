#!/usr/bin/env python3
"""
Hierarchical reduction script v14.

v14 = v13 + checkpoint/resume functionality.

v13 features:
- Beam restart after each weight improvement
- Saves full reduction path for replay

v14 addition:
- Saves checkpoint after each sector completion
- Can resume from checkpoint if interrupted

Usage:
    # Fresh start
    python -u scripts/eval/hierarchical_reduction_v14.py \
        --integral 1,1,1,1,1,1,-3 \
        --output reduction_path.pkl \
        --checkpoint-dir checkpoints/reduction_111111m3

    # Resume from checkpoint
    python -u scripts/eval/hierarchical_reduction_v14.py \
        --resume checkpoints/reduction_111111m3
"""

import sys
import argparse
import time
import pickle
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import torch
from ibp_env import IBPEnvironment, set_prime, is_master, weight, PRIME

# Import from beam_search_classifier_v11 - DO NOT REIMPLEMENT
from beam_search_classifier_v11 import beam_search
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


def reduce_sector_with_restarts(env, model, current_expr, target_sector,
                                 beam_width=20, max_steps_per_restart=300,
                                 device='cpu',
                                 filter_mode='subsector', verbose=True,
                                 n_workers=16, prime=1009):
    """
    Reduce a single sector using beam restart strategy.
    (Same as v13 - no changes needed for checkpointing at sector level)
    """
    expr = dict(current_expr)
    total_steps = 0
    restart_count = 0
    accumulated_path = []

    initial_non_masters = len(get_non_masters_in_sector(expr, target_sector))
    current_weight = max_weight(expr, target_sector)

    while True:
        restart_count += 1

        non_masters = get_non_masters_in_sector(expr, target_sector)
        if not non_masters:
            if verbose:
                print(f"  [Restart {restart_count}] Sector fully reduced!")
            return expr, total_steps, True, accumulated_path

        if verbose:
            print(f"  [Restart {restart_count}] {len(non_masters)} non-masters, max_weight={current_weight[:2]}")

        solution, final_beam, best_weight = beam_search(
            env, model, expr,
            beam_width=beam_width,
            max_steps=max_steps_per_restart,
            device=device,
            verbose=verbose,
            target_sector=target_sector,
            filter_mode=filter_mode,
            use_resolved_subs=True,
            n_workers=n_workers,
            prime=prime,
            patience=None,
            stop_on_weight_improvement=True
        )

        if solution:
            expr = solution.expr
            steps_taken = len(solution.path)
            total_steps += steps_taken
            accumulated_path.extend(solution.path)
            return expr, total_steps, True, accumulated_path

        if not final_beam:
            if verbose:
                print(f"  [Restart {restart_count}] Empty beam - stopping")
            return expr, total_steps, False, accumulated_path

        best_state = final_beam[0]
        steps_taken = len(best_state.path)
        total_steps += steps_taken
        accumulated_path.extend(best_state.path)
        new_weight = max_weight(best_state.expr, target_sector)

        if new_weight < current_weight:
            expr = best_state.expr
            current_weight = new_weight
        else:
            if verbose:
                print(f"  [Restart {restart_count}] No weight improvement after {steps_taken} steps")
            remaining = len(get_non_masters_in_sector(best_state.expr, target_sector))
            if remaining >= len(non_masters):
                return best_state.expr, total_steps, False, accumulated_path
            else:
                expr = best_state.expr


def save_checkpoint(checkpoint_dir, state):
    """Save checkpoint to directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save state as pickle
    checkpoint_file = checkpoint_dir / "checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(state, f)

    # Also save human-readable summary
    summary_file = checkpoint_dir / "checkpoint_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Checkpoint saved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Start integral: {state['start_integral']}\n")
        f.write(f"Iteration: {state['iteration']}\n")
        f.write(f"Total steps: {state['total_steps']}\n")
        f.write(f"Path length: {len(state['full_path'])}\n")
        f.write(f"Sectors completed: {len(state['sector_stats'])}\n")
        f.write(f"Current expr size: {len(state['current_expr'])}\n")
        n_non_masters = sum(1 for i in state['current_expr'] if not is_master(i))
        f.write(f"Non-masters remaining: {n_non_masters}\n")

    print(f"  [CHECKPOINT] Saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_dir):
    """Load checkpoint from directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_file = checkpoint_dir / "checkpoint.pkl"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")

    with open(checkpoint_file, 'rb') as f:
        state = pickle.load(f)

    print(f"Loaded checkpoint from {checkpoint_dir}")
    print(f"  Iteration: {state['iteration']}")
    print(f"  Total steps: {state['total_steps']}")
    print(f"  Path length: {len(state['full_path'])}")
    print(f"  Sectors completed: {len(state['sector_stats'])}")

    return state


def hierarchical_reduction_v14(env, model, start_expr, beam_width=20, max_steps=500,
                                device='cpu', filter_mode='subsector',
                                verbose=True, n_workers=16, prime=1009,
                                checkpoint_dir=None, resume_state=None):
    """
    Hierarchically reduce expression sector by sector with beam restart strategy.

    v14: Supports checkpointing and resume.

    Args:
        checkpoint_dir: Directory to save checkpoints (None to disable)
        resume_state: State dict from previous checkpoint (None for fresh start)

    Returns:
        (final_expr, sector_stats, full_path)
    """
    # Initialize or restore state
    if resume_state is not None:
        current_expr = resume_state['current_expr']
        total_steps = resume_state['total_steps']
        iteration = resume_state['iteration']
        sector_stats = resume_state['sector_stats']
        full_path = resume_state['full_path']
        start_time = time.time() - resume_state.get('elapsed_time', 0)
        print(f"Resuming from iteration {iteration}, {total_steps} steps completed")
    else:
        current_expr = dict(start_expr)
        total_steps = 0
        iteration = 0
        sector_stats = []
        full_path = []
        start_time = time.time()

    start_integral = list(start_expr.keys())[0] if not resume_state else resume_state['start_integral']

    while True:
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

        non_masters = get_non_masters_in_sector(current_expr, target_sector)
        print(f"\nNon-masters in target sector: {len(non_masters)}")
        for nm, coeff in sorted(non_masters.items(), key=lambda x: (-weight(x[0])[0], -weight(x[0])[1]))[:20]:
            print(f"  I{list(nm)} weight={weight(nm)[:2]}")
        if len(non_masters) > 20:
            print(f"  ... and {len(non_masters) - 20} more")

        # Reduce this sector
        final_expr, steps_taken, success, sector_path = reduce_sector_with_restarts(
            env, model, current_expr, target_sector,
            beam_width=beam_width,
            max_steps_per_restart=max_steps,
            device=device,
            filter_mode=filter_mode,
            verbose=verbose,
            n_workers=n_workers,
            prime=prime
        )

        current_expr = final_expr
        total_steps += steps_taken
        full_path.extend(sector_path)

        remaining_non_masters = len(get_non_masters_in_sector(final_expr, target_sector))

        if success:
            print(f"\nSector {list(target_sector)} reduced in {steps_taken} steps")
            sector_stats.append({
                'sector': target_sector,
                'level': level,
                'steps': steps_taken,
                'success': True
            })
        else:
            print(f"\nFAILED to fully reduce sector {list(target_sector)}")
            print(f"  {remaining_non_masters} non-masters remaining after {steps_taken} steps")

            sector_stats.append({
                'sector': target_sector,
                'level': level,
                'steps': steps_taken,
                'success': False,
                'remaining': remaining_non_masters
            })

            if remaining_non_masters >= len(non_masters):
                print(f"  No progress made - stopping.")
                break

        # v14: Save checkpoint after each sector
        if checkpoint_dir is not None:
            elapsed = time.time() - start_time
            checkpoint_state = {
                'start_integral': start_integral,
                'current_expr': current_expr,
                'total_steps': total_steps,
                'iteration': iteration,
                'sector_stats': sector_stats,
                'full_path': full_path,
                'elapsed_time': elapsed,
                'prime': prime,
            }
            save_checkpoint(checkpoint_dir, checkpoint_state)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"HIERARCHICAL REDUCTION v14 COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Total iterations: {iteration}")
    print(f"Total steps: {total_steps}")
    print(f"Path length: {len(full_path)} actions")

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

    return current_expr, sector_stats, full_path


def main():
    parser = argparse.ArgumentParser(description='Hierarchical reduction v14 with checkpoint/resume')
    parser.add_argument('--integral', type=str, default='2,0,2,0,1,1,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--output', type=str,
                        help='Output pickle file for reduction path')
    parser.add_argument('--model-checkpoint', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v5/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Directory to save/load checkpoints')
    parser.add_argument('--resume', type=str,
                        help='Resume from checkpoint directory (overrides other args)')
    parser.add_argument('--beam_width', type=int, default=20,
                        help='Beam width for search')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per beam restart')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on')
    parser.add_argument('--filter_mode', type=str, default='subsector',
                        choices=['subsector', 'higher_only', 'none'],
                        help='Sector filtering mode')
    parser.add_argument('--no-verbose', '-q', action='store_false', dest='verbose',
                        help='Suppress detailed output')
    parser.add_argument('--n_workers', type=int, default=16,
                        help='Number of worker processes')
    args = parser.parse_args()

    print('='*70)
    print('Hierarchical Reduction v14 (Checkpoint/Resume)')
    print('='*70)

    # Handle resume mode
    resume_state = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_state = load_checkpoint(args.resume)
        # Use checkpoint's settings
        args.prime = resume_state['prime']
        args.checkpoint_dir = args.resume
        if 'start_integral' in resume_state:
            args.integral = ','.join(str(x) for x in resume_state['start_integral'])

    print('Innovation: Restart beam search after each weight improvement')
    print('  - Prunes to single best state after each milestone')
    print('  - v14: Checkpoint after each sector for crash recovery')
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
    print(f'\nLoading model from {args.model_checkpoint}...')
    checkpoint = torch.load(args.model_checkpoint, weights_only=False, map_location=args.device)
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

    print(f'\n{"="*70}')
    print(f'Starting hierarchical reduction v14 with {args.n_workers} workers...')
    print(f'{"="*70}')

    final_expr, stats, full_path = hierarchical_reduction_v14(
        env, model, start_expr,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        filter_mode=args.filter_mode,
        verbose=args.verbose,
        n_workers=args.n_workers,
        prime=args.prime,
        checkpoint_dir=args.checkpoint_dir,
        resume_state=resume_state
    )

    # Save final output
    if args.output:
        output_data = {
            'start_integral': indices,
            'final_expr': final_expr,
            'sector_stats': stats,
            'path': full_path,
            'prime': args.prime,
            'args': vars(args)
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f'\n{"="*70}')
        print(f'PATH SAVED to {args.output}')
        print(f'  - {len(full_path)} reduction steps')
        print(f'  - Can be replayed with different primes/kinematics')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()
