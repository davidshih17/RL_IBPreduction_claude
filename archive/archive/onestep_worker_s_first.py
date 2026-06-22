#!/usr/bin/env python3
"""
S-FIRST WORKER: experimental variant of onestep_worker.py that compares
integral weights as (s, r) instead of (r, s).

Rationale: the current model handles high-r low-s integrals fluently but
struggles on low-r high-s ones (deep ISP numerators). Switching the
reduction lex-order to s-first means a worker stops when ANY action lowers
the max s (ISP-weight) in the target sector, even if r grows. This trades
denominator-power budget for ISP reduction, exploiting the part of the
training distribution the model is good at.

Termination: every IBP identity is still constructive — a sufficient
subset of IBPs strictly decreases (s, r) lex too, so an IBP basis exists
under this ordering. Beam search retains its `max_steps` cap, so the
worst case is a non-converging worker that times out (same failure mode
as today), not infinite recursion.

Implementation: monkey-patches `max_weight` in both `beam_search_utils`
and `beam_search` namespaces to return (s, r) tuples instead of (r, s).
Everything else (model, sampling, valid-action enumeration) is unchanged.

Uses the existing beam_search with stop_on_weight_improvement=True.

Usage:
    python onestep_worker.py \
        --integral 3,2,1,3,2,2,-6 \
        --output result.pkl \
        --model-checkpoint checkpoints/best_model.pt \
        --beam_width 20 \
        --max_steps 50 \
        --prime 1009

The output pickle contains:
    - 'success': bool - whether reduction succeeded
    - 'original_integral': tuple - the input integral
    - 'final_expr': dict - the resulting expression (integral -> coeff)
    - 'path': list - the actions taken
    - 'steps': int - number of steps taken
    - 'time': float - wall clock time
    - 'peak_memory_kb': int - peak RSS in KB
"""

import sys
import argparse
import pickle
import time
import resource
from pathlib import Path

# Make sailir.* and the scripts/eval/ siblings importable.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import IBPEnvironment, set_prime, set_paper_masters_only, is_master, weight
from sailir.classifier import IBPActionClassifier
import beam_search as _beam_search_mod
import beam_search_utils as _bsu
from beam_search import beam_search
from beam_search_utils import get_sector_mask
from beam_search_utils import get_non_masters as _get_non_masters


def _max_weight_s_first(expr, target_sector=None):
    """Like beam_search_utils.max_weight but with (s, r) ordering."""
    nms = _get_non_masters(expr, target_sector)
    if not nms:
        return (0, 0)
    return max((weight(k)[1], weight(k)[0]) for k in nms)


# Monkey-patch in BOTH modules. beam_search did `from beam_search_utils
# import max_weight`, so it has its own local reference we need to override.
_bsu.max_weight = _max_weight_s_first
_beam_search_mod.max_weight = _max_weight_s_first
# Also rebind the local name we use below so any local calls match.
max_weight = _max_weight_s_first


def get_non_masters_in_sector(expr, target_sector):
    """Get non-master integrals in the target sector."""
    result = {}
    for integral, coeff in expr.items():
        if coeff == 0:
            continue
        sector = tuple(get_sector_mask(integral))
        if sector != target_sector:
            continue
        if not is_master(integral):
            result[integral] = coeff
    return result


def load_model(checkpoint_path, topology, device='cpu'):
    """Load the classifier model, sized for the given topology."""
    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Reduce single integral by one weight level (v13)')
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory')
    parser.add_argument('--integral', type=str, required=True,
                        help='Integral indices (comma-separated)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=20,
                        help='Beam width for search')
    parser.add_argument('--max_steps', type=int, default=10**15,
                        help='Max steps for beam search (effectively unlimited)')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--paper-masters-only', action='store_true',
                        help='Reduce to paper masters only (no corner integrals)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers for beam search')
    parser.add_argument('--random-target', action='store_true',
                        help='Pick one random tied max-weight target per state instead of all')
    parser.add_argument('--beam-sort', type=str, default='weight',
                        choices=['weight', 'nterms', 'score', 'totalweight', 'mixed'],
                        help='Beam sort key: weight (default), nterms, score, totalweight, mixed')
    args = parser.parse_args()

    start_time = time.time()

    # Setup: configure topology first (sets N_INDICES, etc. in ibp_env), then prime.
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()

    # Parse integral (strip quotes that may be passed from Condor)
    integral_str = args.integral.strip("'\"")
    integral = tuple(int(x) for x in integral_str.split(','))
    target_sector = tuple(get_sector_mask(integral))

    if args.verbose:
        print(f"Loading model from {args.model_checkpoint}")

    model = load_model(args.model_checkpoint, topology, args.device)

    # Start with just this integral
    start_expr = {integral: 1}
    initial_weight = weight(integral)

    if args.verbose:
        print("=" * 60)
        print("S-FIRST onestep worker (max_weight uses (s, r) ordering)")
        print("=" * 60)
        print(f"Reducing integral I{list(integral)}")
        # initial_weight is (r, s, |abs|) from sailir.ibp_env.weight();
        # display both standard and s-first views for clarity.
        print(f"Initial weight (r, s): {initial_weight[:2]}")
        print(f"Initial weight (s, r) [s-first view]: ({initial_weight[1]}, {initial_weight[0]})")
        print(f"Target sector: {list(target_sector)}")

    # Use restart loop like v14 - keep trying until weight improves
    expr = dict(start_expr)
    current_weight = max_weight(expr, target_sector)
    initial_weight_tuple = current_weight
    accumulated_path = []
    # Index into accumulated_path where each beam_search call's contribution
    # begins.  Each beam_search call starts with subs={} internally, so these
    # are the *only* points where the reduction's subs state resets to empty.
    # The replay script needs these explicit boundaries (the sector / weight
    # heuristic over-clears in some cases).
    restart_offsets = []
    total_steps = 0
    restart_count = 0
    success = False
    while True:
        restart_count += 1
        # Get current non-masters count for comparison (like v14)
        non_masters = get_non_masters_in_sector(expr, target_sector)

        if args.verbose:
            print(f"\n  [Restart {restart_count}] current_weight={current_weight}, non_masters={len(non_masters)}")

        solution, final_beam, best_weight = beam_search(
            env, model, expr,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            device=args.device,
            verbose=args.verbose,
            target_sector=target_sector,
            filter_mode='subsector',
            use_resolved_subs=True,
            n_workers=args.n_workers,
            prime=args.prime,
            patience=None,
            stop_on_weight_improvement=True,
            random_target=args.random_target,
            beam_sort=args.beam_sort
        )

        if solution:
            # Fully reduced to masters
            expr = solution.expr
            restart_offsets.append(len(accumulated_path))
            accumulated_path.extend(solution.path)
            total_steps += len(solution.path)
            success = True
            if args.verbose:
                print(f"  Fully reduced to masters!")
            break

        if not final_beam:
            # Empty beam - can't make progress
            if args.verbose:
                print(f"  Empty beam - stopping")
            break

        # Find best state by weight across entire beam (important for mixed mode
        # where the totalweight half may find better weight than the weight half)
        best_state = min(final_beam, key=lambda s: max_weight(s.expr, target_sector))
        new_weight = max_weight(best_state.expr, target_sector)
        steps_taken = len(best_state.path)

        if new_weight < current_weight:
            # Weight improved - accept and stop (goal achieved)
            expr = best_state.expr
            restart_offsets.append(len(accumulated_path))
            accumulated_path.extend(best_state.path)
            total_steps += steps_taken
            current_weight = new_weight
            success = True
            if args.verbose:
                print(f"  Weight improved: {initial_weight_tuple} -> {new_weight}")
            break
        else:
            # No weight improvement - check non-masters like v14
            remaining = len(get_non_masters_in_sector(best_state.expr, target_sector))
            if remaining >= len(non_masters):
                # No progress at all - stop trying
                if args.verbose:
                    print(f"  No weight improvement, no non-masters reduction - stopping")
                break
            else:
                # Non-masters decreased - continue with this state
                expr = best_state.expr
                restart_offsets.append(len(accumulated_path))
                accumulated_path.extend(best_state.path)
                total_steps += steps_taken
                if args.verbose:
                    print(f"  No weight improvement, but non-masters {len(non_masters)} -> {remaining}, continuing...")

    elapsed = time.time() - start_time

    # Final check: only succeed if weight actually improved
    final_weight = max_weight(expr, target_sector)
    if final_weight < initial_weight_tuple:
        final_expr = expr
        path = accumulated_path
        steps = total_steps
        success = True
    else:
        # Failed to improve weight - return original
        final_expr = start_expr
        path = []
        steps = 0
        success = False
        restart_offsets = []
        if args.verbose:
            print(f"\nFailed to improve weight after {restart_count} restarts")

    # Get peak memory usage (in KB on Linux)
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Save results
    result = {
        'success': success,
        'original_integral': integral,
        'final_expr': final_expr,
        'path': path,
        # Indices in `path` where each beam_search call's contribution begins.
        # Subs are empty at the start of each such call -- replay must clear
        # subs at exactly these positions.
        'restart_offsets': restart_offsets,
        'steps': steps,
        'time': elapsed,
        'prime': args.prime,
        'peak_memory_kb': peak_rss_kb
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    if args.verbose:
        status = "SUCCESS" if success else "FAILED"
        per_cpu_mb = peak_rss_kb / 1024 / args.n_workers if args.n_workers > 1 else peak_rss_kb / 1024
        print(f"\n{status} in {elapsed:.2f}s, peak memory: {peak_rss_kb/1024:.1f} MB (per-CPU: {per_cpu_mb:.1f} MB, n_workers={args.n_workers})")
        print(f"Output saved to {args.output}")


if __name__ == '__main__':
    main()
