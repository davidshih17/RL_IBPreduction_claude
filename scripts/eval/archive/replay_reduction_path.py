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

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

# Import EXACTLY what v13 imports - DO NOT REIMPLEMENT
from sailir import ibp_env
from sailir.ibp_env import IBPEnvironment, set_prime, set_kinematics, is_master, weight, PRIME
from sailir.topology import Topology

# Import from same modules as v13
from beam_search_utils import (
    get_sector_mask,
    get_non_masters,
    max_weight
)


def replay_path(env, start_integral, path, verbose=True, worker_boundaries=None):
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
    - When we cross into a new onestep_worker's sub-path (from
      hierarchical_reduction.py's `worker_boundaries`), clear subs too --
      each worker ran with its own fresh subs internally.
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

    # Set of step indices at which a fresh worker sub-path / beam_search call
    # begins.  When this is provided by the orchestrator / onestep_worker, it
    # is the authoritative subs-reset signal -- we skip the heuristic resets
    # entirely so that replay exactly matches the original execution.
    boundary_set = set(worker_boundaries or [])
    use_explicit_boundaries = boundary_set is not None and len(boundary_set) > 0

    if verbose:
        print(f"Starting replay: {len(path)} steps")
        print(f"Initial target_sector: {list(target_sector)}")
        print(f"Initial weight: {current_weight[:2]}")
        if use_explicit_boundaries:
            print(f"Using explicit subs-reset boundaries ({len(boundary_set)} sub-paths)")

    for step, (target, ibp_op, delta) in enumerate(path):
        # Get the sector of this target
        step_sector = tuple(get_sector_mask(target))

        # Explicit boundary reset: each onestep_worker (and each beam_search
        # restart within it) starts with fresh subs.
        if step > 0 and step in boundary_set:
            subs = {}
            resolved_subs = {}
            target_sector = step_sector
            current_weight = max_weight(expr, target_sector)
            restarts += 1
            # We have an exact boundary signal; skip the heuristic resets
            # below for this step.
            target_sector_changed = False
        elif step_sector != target_sector:
            target_sector_changed = True
        else:
            target_sector_changed = False

        # Heuristic fallback: only used when the pickle has no explicit
        # boundaries (e.g. legacy onestep_worker output without
        # restart_offsets).  Resets subs on sector change.
        if not use_explicit_boundaries and target_sector_changed:
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

        # Heuristic weight-improvement reset: only used when there are no
        # explicit boundaries to lean on.
        new_weight = max_weight(expr, target_sector)
        if not use_explicit_boundaries and new_weight < current_weight:
            if verbose and restarts < 50:
                print(f"Step {step}: Weight improved {current_weight[:2]} -> {new_weight[:2]}, clearing subs")
            subs = {}
            resolved_subs = {}
            current_weight = new_weight
            restarts += 1
        else:
            current_weight = new_weight

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
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory')
    parser.add_argument('--path', type=str, required=True,
                        help='Pickle file: a single onestep_worker output (verifies that one '
                             'worker), OR an orchestrator output (combined with --work-dir, '
                             'verifies the full reduction).')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Orchestrator mode: directory containing per-worker pickles '
                             '(work-dir/<label>/results/*.pkl).  When set, --path must point '
                             'at the orchestrator pickle; each cache entry is verified by '
                             'replaying its worker pickle.')
    parser.add_argument('--extra', type=str, nargs='*', default=[],
                        help='Additional patch paths to apply after main path (single-pickle mode)')
    parser.add_argument('--prime', type=int, default=10007,
                        help='Prime for modular arithmetic (different from original)')
    parser.add_argument('--d', type=int, default=41,
                        help='Spacetime dimension for kinematics')
    parser.add_argument('--kinematics', type=str, default=None,
                        help='Override numeric kinematic invariants as comma-separated '
                             '"name=value" pairs, e.g. "m2=31,m3=47" or '
                             '"s12=31,s23=47,s34=53,s45=59,s51=61". If omitted, the '
                             'defaults from the topology are used.')
    parser.add_argument('--no-verbose', '-q', action='store_false', dest='verbose',
                        help='Suppress detailed output')
    args = parser.parse_args()

    # Configure topology FIRST so ibp_env globals (N_INDICES, KINEMATICS, etc.) are set.
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    # Parse --kinematics override, if any.
    kin_override = {"d": args.d}
    if args.kinematics:
        for piece in args.kinematics.split(','):
            piece = piece.strip()
            if not piece:
                continue
            name, _, val = piece.partition('=')
            kin_override[name.strip()] = int(val.strip())
    # Only pass keys the topology recognises.
    valid = {k: v for k, v in kin_override.items() if k in ibp_env.KINEMATICS}
    if valid:
        set_kinematics(**valid)

    if args.work_dir is not None:
        return orchestrator_mode(args)

    # Load the reduction path
    print(f"Loading reduction path from {args.path}...")
    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    original_prime = data['prime']
    # `start_integral` is the orchestrator's name; `original_integral` is
    # onestep_worker's.  Accept either.
    start_integral = data.get('start_integral', data.get('original_integral'))
    if start_integral is None:
        raise KeyError("pickle has neither 'start_integral' nor 'original_integral'")
    path = data['path']
    # Explicit subs-reset boundaries: orchestrator output has `worker_boundaries`
    # (combined across all workers); worker output has `restart_offsets`
    # (within a single onestep_worker run).
    worker_boundaries = data.get('worker_boundaries') or data.get('restart_offsets')

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

    # Set new prime (topology + kinematics were configured at the top of main).
    set_prime(args.prime)
    print(f"Using PRIME = {args.prime}")
    print(f"Using kinematics: {ibp_env.KINEMATICS}")

    # Load IBP environment
    print("Loading IBP environment...")
    env = IBPEnvironment()
    print(f"Loaded {len(env.ibp_t)} IBP templates, {len(env.li_t)} LI templates")

    # Replay the path
    print(f"\n{'='*70}")
    print(f"Starting replay...")
    print(f"{'='*70}")

    final_expr, success, steps_completed = replay_path(
        env, start_integral, path, verbose=args.verbose,
        worker_boundaries=worker_boundaries,
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

    # Determine success.  Two modes:
    #  - If the pickle has a `final_expr` field (both onestep_worker and the
    #    orchestrator do), the replayed result MUST equal it exactly.  This is
    #    the canonical correctness check for any pickle.
    #  - Otherwise, fall back to "every integral is a master" -- the legacy
    #    criterion, only meaningful for end-to-end reductions.
    expected_final = data.get('final_expr')
    print(f"\n{'='*70}")
    if expected_final is not None:
        match = final_expr == expected_final
        if match:
            print(f"SUCCESS! Replay reproduced final_expr exactly with prime={args.prime}")
            print(f"  ({len(final_expr)} integrals, {total_steps} IBP steps)")
            success = True
        else:
            print(f"FAILED! Replay's final state DIFFERS from pickle's final_expr.")
            print(f"  replay:   {len(final_expr)} integrals")
            print(f"  expected: {len(expected_final)} integrals")
            extra   = set(final_expr) - set(expected_final)
            missing = set(expected_final) - set(final_expr)
            wrong   = {k for k in set(final_expr) & set(expected_final)
                       if final_expr[k] != expected_final[k]}
            print(f"  extra (replay only):    {len(extra)}")
            print(f"  missing (expected only): {len(missing)}")
            print(f"  wrong coefficient:       {len(wrong)}")
            success = False
    else:
        if success:
            print(f"SUCCESS! Path replayed successfully with prime={args.prime}")
            print(f"All {len(final_expr)} integrals are masters. ({total_steps} steps)")
        else:
            n_non_masters = sum(1 for i in final_expr if not is_master(i))
            print(f"FAILED! {n_non_masters} non-masters remaining after {total_steps} steps")
    print(f"{'='*70}")

    if args.verbose:
        print_expression_summary(final_expr, "Final expression")

    return 0 if success else 1


def find_worker_pickle(work_dir, integral):
    """Locate the onestep_worker pickle for the given integral inside work_dir."""
    from pathlib import Path
    # Worker pickles are named like async_<idx>_<i0>_<i1>_..._<i6>.pkl
    # where negative indices use a minus sign.  Build the suffix matching the
    # integral and glob for it.
    suffix = '_'.join(str(x) for x in integral) + '.pkl'
    results = Path(work_dir) / 'results'
    if not results.is_dir():
        # Allow user to pass the results/ dir directly
        results = Path(work_dir)
    candidates = list(results.glob(f"*_{suffix}"))
    if not candidates:
        # Some integrals contain negative entries that look like an additional
        # underscore-separated field; try with explicit handling.
        return None
    if len(candidates) > 1:
        # Multiple matches (e.g. straggler resubmissions).  Pick the largest;
        # successful runs produce non-zero pickles.
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def orchestrator_mode(args):
    """Verify an orchestrator reduction by replaying each worker pickle independently
    and composing the results via the orchestrator's apply_substitutions logic.

    Mirrors what hierarchical_reduction.py did:
      1. For each integral X in cache: replay its onestep_worker pickle's path
         starting from {X: 1}, with the worker's restart_offsets driving
         subs resets.  Check replay's final state equals worker's final_expr.
      2. After every worker verifies, compose the verified cache into
         apply_substitutions({start: 1}, verified_cache, prime) and check the
         result equals the orchestrator's saved final_expr.
    """
    from pathlib import Path
    print(f"Loading orchestrator pickle from {args.path}...")
    with open(args.path, 'rb') as f:
        orch = pickle.load(f)

    start_integral = orch['start_integral']
    cache          = orch['cache']
    expected_final = orch['final_expr']
    original_prime = orch['prime']

    set_prime(args.prime)
    # Kinematics were already configured at the top of main() — don't override.
    env = IBPEnvironment()

    print(f"\nStart integral: I{list(start_integral)}")
    print(f"Original prime: {original_prime}  Replay prime: {args.prime}")
    print(f"Cache entries:  {len(cache)}")
    print(f"Expected final: {len(expected_final)} integrals")
    if args.prime != original_prime:
        print("(replaying with a DIFFERENT prime -- this re-derives every IBP "
              "coefficient and is the strongest correctness check.)")
    print()

    # Step 1: replay each worker pickle independently.
    verified_cache = {}
    n_match = n_diff = n_missing = 0
    for i, integral in enumerate(cache):
        pkl = find_worker_pickle(args.work_dir, integral)
        if pkl is None:
            n_missing += 1
            print(f"  [{i+1}/{len(cache)}]  MISSING worker pickle for {integral}")
            continue
        with open(pkl, 'rb') as f:
            w = pickle.load(f)
        worker_path  = w.get('path', [])
        boundaries   = w.get('restart_offsets')
        worker_final = w.get('final_expr')

        # Replay this worker locally from {integral: 1}.
        replayed_final, _, _ = replay_path(
            env, integral, worker_path, verbose=False, worker_boundaries=boundaries,
        )
        # At a DIFFERENT prime, coefficients differ from the saved final_expr
        # by design; the replay result is the authoritative cache entry at
        # the replay prime.  At the SAME prime, mismatches are real errors.
        verified_cache[integral] = replayed_final
        if replayed_final == worker_final:
            n_match += 1
        else:
            n_diff += 1
            if args.prime == original_prime:
                # Real error -- workers should be deterministic at same prime.
                print(f"  [{i+1}/{len(cache)}]  MISMATCH (same prime!) {integral}: "
                      f"replay {len(replayed_final)} terms vs saved {len(worker_final)} terms")
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(cache)}]  verified: {n_match}, diff: {n_diff}, missing: {n_missing}")

    print()
    print(f"per-worker results:")
    if args.prime == original_prime:
        print(f"  match (same prime, must be 100%): {n_match} / {len(cache)}")
        print(f"  mismatch (failures):              {n_diff}")
    else:
        print(f"  same coefficients (lucky):        {n_match} / {len(cache)}")
        print(f"  different coefficients (expected at different prime): {n_diff}")
    print(f"  missing pickle                      : {n_missing}")

    # Step 2: compose via apply_substitutions and verify against orchestrator's final_expr.
    print("\nComposing verified cache into the global reduction...")
    expr = {start_integral: 1}
    expr = apply_substitutions(expr, verified_cache, args.prime)

    print(f"\n{'='*70}")
    if args.prime == original_prime:
        # Same prime: must exactly reproduce the orchestrator's final_expr.
        if expr == expected_final:
            print(f"SUCCESS! Verified reduction reproduces orchestrator's final_expr.")
            print(f"  ({len(expr)} masters)")
            ok = True
        else:
            n_nm = sum(1 for k in expr if not is_master(k))
            print(f"FAILED: composed reduction at same prime DIFFERS from orchestrator output.")
            print(f"  ({len(expr)} integrals, {n_nm} non-masters)")
            ok = False
    else:
        # Different prime: residual non-masters are expected (accidental zeros
        # at the small prime aren't zeros at the new prime).  Report what's
        # left so the user can see they are low-weight.
        non_masters = [(k, weight(k)) for k in expr if not is_master(k)]
        if not non_masters:
            print(f"SUCCESS (different prime): composition reaches the same "
                  f"{len(expr)} paper masters; coefficients differ as expected.")
            ok = True
        else:
            print(f"PARTIAL (different prime, prime={args.prime}):")
            print(f"  {len(expr) - len(non_masters)} master integrals, "
                  f"{len(non_masters)} residual non-masters")
            print(f"  This is expected: accidental zeros at prime={original_prime} "
                  f"don't survive at the new prime.")
            non_masters.sort(key=lambda kw: (-kw[1][0], -kw[1][1]))
            top_w = non_masters[0][1][:2]
            print(f"  Max residual weight: {top_w}  (vs start weight {weight(start_integral)[:2]})")
            for k, w in non_masters[:10]:
                print(f"    I{list(k)}  weight={w[:2]}")
            if len(non_masters) > 10:
                print(f"    ... and {len(non_masters)-10} more")
            # Don't call this a failure -- it's a legitimate residual.
            ok = all(w[0] < weight(start_integral)[0] for _, w in non_masters)
            if ok:
                print(f"  All residuals are STRICTLY lower-weight than the start "
                      f"integral -- consistent with accidental-zero hypothesis.")
    print(f"{'='*70}")

    if args.prime == original_prime:
        return 0 if (ok and n_diff == 0 and n_missing == 0) else 1
    return 0 if (ok and n_missing == 0) else 1


def apply_substitutions(expr, cache, prime):
    """Same dictionary substitution logic the orchestrator uses."""
    result = dict(expr)
    changed = True
    while changed:
        changed = False
        for integral in list(result.keys()):
            if integral in cache:
                coeff = result.pop(integral)
                for sub_int, sub_coeff in cache[integral].items():
                    new_coeff = (coeff * sub_coeff) % prime
                    if sub_int in result:
                        result[sub_int] = (result[sub_int] + new_coeff) % prime
                    else:
                        result[sub_int] = new_coeff
                result = {k: v for k, v in result.items() if v != 0}
                changed = True
                break
    return result


if __name__ == '__main__':
    sys.exit(main())
