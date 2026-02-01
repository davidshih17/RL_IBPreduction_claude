#!/usr/bin/env python3
"""
Replay a saved reduction path with different prime/kinematics.

Uses EXACTLY the same code as v13 - imports from ibp_env and beam_search.
Mirrors v13's structure: sector changes and weight improvements clear subs.

Usage:
    python -u scripts/eval/replay_reduction_path.py \
        --path results/reduction_111111m3.pkl \
        --prime 10007 \
        -v
"""

import sys
import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

# Import EXACTLY what v13 imports - DO NOT REIMPLEMENT
from ibp_env import IBPEnvironment, set_prime, set_kinematics, is_master, weight, PRIME

# Import from same modules as v13
from beam_search_classifier_v4_target import (
    get_sector_mask,
    get_non_masters,
    max_weight
)


def replay_path(env, start_integral, path, verbose=True):
    """
    Replay a reduction path using EXACTLY the same logic as v13.

    In v13:
    - reduce_sector_with_restarts processes one sector at a time
    - target_sector is the sector currently being reduced
    - beam_search starts with subs={} and accumulates until weight improves
    - When weight improves, beam_search is called again with fresh subs={}
    - When moving to a new sector, reduce_sector_with_restarts is called fresh

    Here we mirror this:
    - target_sector = sector of the current target being eliminated
    - When target's sector changes -> new sector, clear subs
    - When weight improves -> restart, clear subs
    """
    expr = {start_integral: 1}
    subs = {}
    resolved_subs = {}

    start_time = time.time()
    restarts = 0

    # Determine initial target_sector from first step
    if not path:
        return expr, True, 0

    # target_sector is the sector of the integral currently being reduced
    first_target = path[0][0]
    target_sector = tuple(get_sector_mask(first_target))
    current_weight = max_weight(expr, target_sector)

    if verbose:
        print(f"Starting replay: {len(path)} steps")
        print(f"Initial target_sector: {list(target_sector)}")
        print(f"Initial weight: {current_weight[:2]}")

    for step, (target, ibp_op, delta) in enumerate(path):
        # Get the sector of this target
        step_sector = tuple(get_sector_mask(target))

        # Check if we've moved to a new sector
        # This mirrors v13: each new sector calls reduce_sector_with_restarts
        # which calls beam_search with fresh subs={}
        if step_sector != target_sector:
            if verbose:
                print(f"Step {step}: Sector change {list(target_sector)} -> {list(step_sector)}, clearing subs")
            subs = {}
            resolved_subs = {}
            target_sector = step_sector
            current_weight = max_weight(expr, target_sector)
            restarts += 1

        # Apply the action using EXACTLY the same function as v13/beam_search
        new_expr, new_subs, new_resolved_subs, success = env.apply_action_resolved(
            dict(expr), dict(subs), resolved_subs, target, ibp_op, delta
        )

        if not success:
            if verbose:
                print(f"Step {step}: FAILED - apply_action_resolved returned False")
                print(f"  target={list(target)}, ibp_op={ibp_op}, delta={delta}")
                print(f"  subs has {len(subs)} entries")
            return expr, False, step

        # Update state
        expr = new_expr
        subs = new_subs
        resolved_subs = new_resolved_subs

        # Check if weight improved within this sector
        # This mirrors v13: when weight improves, beam_search returns and is
        # called again with fresh subs={}
        new_weight = max_weight(expr, target_sector)
        if new_weight < current_weight:
            if verbose and restarts < 50:
                print(f"Step {step}: Weight improved {current_weight[:2]} -> {new_weight[:2]}, clearing subs")
            subs = {}
            resolved_subs = {}
            current_weight = new_weight
            restarts += 1

        # Progress logging
        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            n_integrals = len(expr)
            n_non_masters = sum(1 for i in expr if not is_master(i))
            print(f"Step {step + 1}/{len(path)}: {n_integrals} integrals, {n_non_masters} non-masters, {restarts} restarts, {elapsed:.1f}s")

    elapsed = time.time() - start_time

    # Check final result
    n_non_masters = sum(1 for i in expr if not is_master(i))

    if verbose:
        print(f"\nReplay complete: {len(path)} steps in {elapsed:.1f}s")
        print(f"Restarts: {restarts}")
        print(f"Final expression: {len(expr)} integrals, {n_non_masters} non-masters")

    success = (n_non_masters == 0)
    return expr, success, len(path)


def print_expression_summary(expr, title="Expression"):
    """Print summary of expression."""
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
    for sector in sorted(by_sector.keys(), key=lambda s: (-sum(s), -sum(b << i for i, b in enumerate(s)))):
        level = sum(sector)
        sint = sum(b << i for i, b in enumerate(sector))
        integrals = by_sector[sector]
        n_masters = sum(1 for _, _, is_m, _ in integrals if is_m)
        n_non_masters = len(integrals) - n_masters
        print(f"  Sector {list(sector)} (level {level}, int {sint}): {len(integrals)} integrals ({n_masters} masters, {n_non_masters} non-masters)")
        for integral, coeff, is_m, w in sorted(integrals, key=lambda x: (-x[3][0], -x[3][1]))[:5]:
            tag = "master" if is_m else "NON-MASTER"
            print(f"    I{list(integral)} coeff={coeff} weight={w[:2]} [{tag}]")
        if len(integrals) > 5:
            print(f"    ... and {len(integrals) - 5} more")


def main():
    parser = argparse.ArgumentParser(description='Replay reduction path with different prime')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to pickle file with main reduction path')
    parser.add_argument('--extra', type=str, nargs='*', default=[],
                        help='Additional patch paths to apply after main path')
    parser.add_argument('--prime', type=int, default=10007,
                        help='Prime for modular arithmetic (different from original)')
    parser.add_argument('--d', type=int, default=41,
                        help='Spacetime dimension for kinematics')
    parser.add_argument('--m1', type=int, default=1,
                        help='Mass m1 for kinematics')
    parser.add_argument('--m2', type=int, default=31,
                        help='Mass m2 for kinematics')
    parser.add_argument('--m3', type=int, default=47,
                        help='Mass m3 for kinematics')
    parser.add_argument('--no-verbose', '-q', action='store_false', dest='verbose',
                        help='Suppress detailed output')
    args = parser.parse_args()

    # Load the reduction path
    print(f"Loading reduction path from {args.path}...")
    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    original_prime = data['prime']
    start_integral = data['start_integral']
    path = data['path']

    print(f"\n{'='*70}")
    print(f"REPLAY REDUCTION PATH")
    print(f"{'='*70}")
    print(f"Start integral: I{list(start_integral)}")
    print(f"Path length: {len(path)} steps")
    print(f"Original prime: {original_prime}")
    print(f"Replay prime: {args.prime}")
    print(f"{'='*70}\n")

    if args.prime == original_prime:
        print("WARNING: Replay prime is same as original - this just verifies the path is consistent")

    # Set new prime and kinematics
    set_prime(args.prime)
    set_kinematics(d=args.d, m1=args.m1, m2=args.m2, m3=args.m3)
    print(f"Using PRIME = {args.prime}")
    print(f"Using kinematics: d={args.d}, m1={args.m1}, m2={args.m2}, m3={args.m3}")

    # Load IBP environment
    print("Loading IBP environment...")
    env = IBPEnvironment()
    print(f"Loaded {len(env.ibp_t)} IBP templates, {len(env.li_t)} LI templates")

    # Replay the path
    print(f"\n{'='*70}")
    print(f"Starting replay...")
    print(f"{'='*70}")

    final_expr, success, steps_completed = replay_path(
        env, start_integral, path, verbose=args.verbose
    )

    total_steps = steps_completed

    # Apply extra paths if provided and there are remaining non-masters
    if args.extra and not success:
        print(f"\n{'='*70}")
        print(f"Applying {len(args.extra)} extra patch path(s)...")
        print(f"{'='*70}")

        for extra_file in args.extra:
            n_non_masters = sum(1 for i in final_expr if not is_master(i))
            if n_non_masters == 0:
                break

            print(f"\nLoading extra path from {extra_file}...")
            with open(extra_file, 'rb') as f:
                extra_data = pickle.load(f)

            extra_path = extra_data['path']
            extra_start = extra_data['start_integral']
            print(f"  Extra path for I{list(extra_start)}: {len(extra_path)} steps")

            # Check if this extra path's start integral is in our expression
            if extra_start not in final_expr:
                print(f"  Skipping - integral not in current expression")
                continue

            # Apply the extra path starting from current expression
            print(f"  Applying {len(extra_path)} steps...")
            expr = final_expr
            subs = {}
            resolved_subs = {}

            # Determine target sector from first step
            if extra_path:
                target_sector = tuple(get_sector_mask(extra_path[0][0]))
                current_weight = max_weight(expr, target_sector)

            for step, (target, ibp_op, delta) in enumerate(extra_path):
                step_sector = tuple(get_sector_mask(target))

                if step_sector != target_sector:
                    subs = {}
                    resolved_subs = {}
                    target_sector = step_sector
                    current_weight = max_weight(expr, target_sector)

                new_expr, new_subs, new_resolved_subs, step_success = env.apply_action_resolved(
                    dict(expr), dict(subs), resolved_subs, target, ibp_op, delta
                )

                if not step_success:
                    print(f"  Step {step}: FAILED")
                    break

                expr = new_expr
                subs = new_subs
                resolved_subs = new_resolved_subs

                new_weight = max_weight(expr, target_sector)
                if new_weight < current_weight:
                    subs = {}
                    resolved_subs = {}
                    current_weight = new_weight

                total_steps += 1

            final_expr = expr

        # Recheck success after extra paths
        n_non_masters = sum(1 for i in final_expr if not is_master(i))
        success = (n_non_masters == 0)

    # Print results
    print(f"\n{'='*70}")
    if success:
        print(f"SUCCESS! Path replayed successfully with prime={args.prime}")
        print(f"All {len(final_expr)} integrals are masters.")
        print(f"Total steps: {total_steps}")
    else:
        n_non_masters = sum(1 for i in final_expr if not is_master(i))
        print(f"FAILED! {n_non_masters} non-masters remaining after {total_steps} steps")
        print(f"This could mean:")
        print(f"  1. Need additional patch paths for uncovered sectors")
        print(f"  2. The prime causes a division by zero (unlucky prime)")
    print(f"{'='*70}")

    if args.verbose:
        print_expression_summary(final_expr, "Final expression")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
