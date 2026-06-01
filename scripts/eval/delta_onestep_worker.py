#!/usr/bin/env python3
"""Delta-tracking serial worker (mirrors onestep_worker.py).

Drives sailir.delta_beam_search.beam_search_delta for a single integral.
Single-process (n_workers=1), so no Condor pool plumbing.

Output pickle schema matches onestep_worker.py so downstream scripts
(replay/verify/etc.) work unchanged.
"""

import sys
import argparse
import pickle
import time
import resource
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root
sys.path.insert(0, str(_HERE.parent))                # scripts/eval/ siblings

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, set_paper_masters_only, weight,
)
from sailir.classifier import IBPActionClassifier
from sailir.delta_beam_search import beam_search_delta
from beam_search_utils import get_sector_mask, max_weight
from onestep_worker import get_non_masters_in_sector


def load_model(checkpoint_path, topology, device='cpu'):
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
    parser = argparse.ArgumentParser(description='Delta-tracking serial worker')
    parser.add_argument('--topology', type=str, required=True)
    parser.add_argument('--integral', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--beam_width', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=10**15)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--paper-masters-only', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of CPU threads for torch BLAS ops. '
                             'P2 (model inference) speeds up roughly linearly '
                             'with this if request_cpus matches. Default 1.')
    args = parser.parse_args()

    # Set torch thread count BEFORE loading model. BLAS ops in the
    # model forward pass (transformer attention, linear layers) will
    # then use this many threads.
    import torch as _t
    _t.set_num_threads(args.n_threads)
    _t.set_num_interop_threads(args.n_threads)
    print(f'torch: n_threads={_t.get_num_threads()} '
          f'interop_threads={_t.get_num_interop_threads()}', flush=True)

    start_time = time.time()

    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()

    integral_str = args.integral.strip("'\"")
    integral = tuple(int(x) for x in integral_str.split(','))
    target_sector = tuple(get_sector_mask(integral))

    if args.verbose:
        print(f'Loading model from {args.model_checkpoint}', flush=True)
    model = load_model(args.model_checkpoint, topology, args.device)

    start_expr = {integral: 1}
    initial_weight = weight(integral)
    if args.verbose:
        print(f'Reducing I{list(integral)}', flush=True)
        print(f'Initial weight: {initial_weight[:2]}', flush=True)
        print(f'Target sector: {list(target_sector)}', flush=True)

    # Restart loop mirrors onestep_worker.py — beam_search runs until
    # weight improves (or empty beam), and we accept the best state.
    expr = dict(start_expr)
    current_weight = max_weight(expr, target_sector)
    initial_weight_tuple = current_weight
    accumulated_path = []
    restart_offsets = []
    total_steps = 0
    restart_count = 0
    success = False

    while True:
        restart_count += 1
        non_masters = get_non_masters_in_sector(expr, target_sector)
        if args.verbose:
            print(f'\n  [Restart {restart_count}] current_weight={current_weight} '
                  f'non_masters={len(non_masters)}', flush=True)

        this_call_start_expr = dict(expr)
        solution, final_beam, best_weight = beam_search_delta(
            env, model, expr,
            target_sector=target_sector,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            prime=args.prime,
            verbose=args.verbose,
            stop_on_weight_improvement=True,
            filter_mode='subsector',
            device=args.device,
        )

        def _full_expr(state):
            # Same Option-F reconstruct as production.
            return env.replay_path_to_full_expr(this_call_start_expr, state.path)

        if solution:
            expr = _full_expr(solution)
            restart_offsets.append(len(accumulated_path))
            accumulated_path.extend(solution.path)
            total_steps += solution.path_len
            success = True
            if args.verbose:
                print('  Fully reduced to masters!', flush=True)
            break

        if not final_beam:
            if args.verbose:
                print('  Empty beam - stopping', flush=True)
            break

        best_state = min(final_beam, key=lambda s: max_weight(s.expr, target_sector)
                         if s.expr is not None else (10**9, 10**9))
        # If the best survivor's expr was nulled by cleanup (shouldn't happen for
        # current-beam survivors — only their PARENTS get nulled — but defensively):
        if best_state.expr is None:
            if args.verbose:
                print('  Best survivor has no expr (cleanup bug?), stopping', flush=True)
            break

        new_weight = max_weight(best_state.expr, target_sector)
        steps_taken = best_state.path_len

        if new_weight < current_weight:
            expr = _full_expr(best_state)
            restart_offsets.append(len(accumulated_path))
            accumulated_path.extend(best_state.path)
            total_steps += steps_taken
            current_weight = new_weight
            success = True
            if args.verbose:
                print(f'  Weight improved: {initial_weight_tuple} -> {new_weight}', flush=True)
            break
        else:
            remaining = len(get_non_masters_in_sector(best_state.expr, target_sector))
            if remaining >= len(non_masters):
                if args.verbose:
                    print('  No progress - stopping', flush=True)
                break
            expr = _full_expr(best_state)
            restart_offsets.append(len(accumulated_path))
            accumulated_path.extend(best_state.path)
            total_steps += steps_taken
            if args.verbose:
                print(f'  Non-masters reduced {len(non_masters)} -> {remaining}, '
                      f'continuing...', flush=True)

    elapsed = time.time() - start_time

    final_weight = max_weight(expr, target_sector)
    if final_weight < initial_weight_tuple:
        final_expr = expr
        path = accumulated_path
        steps = total_steps
        success = True
    else:
        final_expr = start_expr
        path = []
        steps = 0
        success = False
        restart_offsets = []
        if args.verbose:
            print(f'\nFailed to improve weight after {restart_count} restarts', flush=True)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        'success': success,
        'original_integral': integral,
        'final_expr': final_expr,
        'path': path,
        'restart_offsets': restart_offsets,
        'steps': steps,
        'time': elapsed,
        'prime': args.prime,
        'peak_memory_kb': peak_rss_kb,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    if args.verbose:
        status = 'SUCCESS' if success else 'FAILED'
        print(f'\n{status} in {elapsed:.2f}s, '
              f'peak memory: {peak_rss_kb/1024:.1f} MB, '
              f'steps={steps}', flush=True)
        print(f'Output saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
