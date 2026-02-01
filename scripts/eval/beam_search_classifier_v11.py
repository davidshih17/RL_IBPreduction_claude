#!/usr/bin/env python3
"""
Beam search reduction using classifier v5 with full parallelization.

v11 changes from v7:
- Remove max_terms=200 limit (model sees full expression)
- Take LAST 50 subs instead of first 50 (most recent = most relevant for current weight)

Usage:
    python -u scripts/eval/beam_search_classifier_v11.py --beam_width 10 --max_steps 100 --n_workers 16
"""

import sys
import argparse
from pathlib import Path
from collections import namedtuple
import multiprocessing as mp
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

import numpy as np
import torch
from ibp_env import (
    IBPEnvironment, set_prime, filter_top_sector, is_master, weight, resolve_subs,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector
)
from classifier_v5 import IBPActionClassifierV5

# Reuse core functions from v4
from beam_search_classifier_v4_target import (
    get_sector_mask,
    filter_to_sector,
    get_non_masters,
    max_weight
)

# Our own State that includes resolved_subs for optimization
State = namedtuple('State', ['expr', 'subs', 'resolved_subs', 'score', 'path', 'n_non_masters'])

MAX_REPLACEMENT_TERMS = 20  # Must match training

# Global worker environment (initialized once per worker process)
_worker_env = None


def init_worker(prime):
    """Initialize worker process with its own IBPEnvironment."""
    global _worker_env
    set_prime(prime)
    _worker_env = IBPEnvironment()


def process_state_tasks(args):
    """Worker function: process all targets for a single state.

    Args:
        args: tuple of (state_idx, state_subs, state_resolved_subs, targets_info, target_sector, filter_mode)
              where targets_info is [(task_idx, target), ...]

    Returns:
        List of (task_idx, valid_actions) tuples
    """
    global _worker_env

    state_idx, state_subs, state_resolved_subs, targets_info, target_sector, filter_mode = args

    # Filter subs for this state
    if target_sector is not None:
        filtered_subs = filter_subs_to_exact_sector(state_subs, target_sector)
        filtered_resolved = filter_resolved_subs_to_exact_sector(state_resolved_subs, target_sector)
    else:
        filtered_subs = state_subs
        filtered_resolved = state_resolved_subs

    # Compute indirect cache ONCE for this state
    indirect_cache = _worker_env.compute_indirect_cache(filtered_subs, filtered_resolved)

    # Get valid actions for each target
    results = []
    for task_idx, target in targets_info:
        valid_actions = _worker_env.get_valid_actions_with_cache(
            target, indirect_cache, filtered_subs, filtered_resolved,
            filter_mode=filter_mode, verbose_timing=False
        )
        results.append((task_idx, valid_actions))

    return results


def apply_actions_worker(args):
    """Worker function: apply a batch of actions and return candidate states.

    Args:
        args: tuple of (action_batch, target_sector)
              action_batch is list of (expr, subs, resolved_subs, target, ibp_op, delta, action_prob, path, score)

    Returns:
        List of (new_expr, new_subs, new_resolved_subs, new_path, new_score, n_non_masters) or None if failed
    """
    global _worker_env

    action_batch, target_sector = args
    results = []

    for expr, subs, resolved_subs, target, ibp_op, delta, action_prob, path, score in action_batch:
        new_expr, new_subs, new_resolved_subs, success = _worker_env.apply_action_resolved(
            dict(expr), dict(subs), resolved_subs, target, ibp_op, delta
        )

        if not success:
            results.append(None)
            continue

        # Count non-masters in target sector
        n_non_masters = 0
        for integral in new_expr.keys():
            if target_sector is not None:
                sector = tuple(1 if integral[i] > 0 else 0 for i in range(6))
                if sector != target_sector:
                    continue
            if not is_master(integral):
                n_non_masters += 1

        new_path = path + [(target, ibp_op, delta)]
        import math
        new_score = score + math.log(action_prob + 1e-10)

        results.append((new_expr, new_subs, new_resolved_subs, new_path, new_score, n_non_masters))

    return results


def prepare_batched_input_v5(batch_data, device):
    """
    Prepare batched model input for V5 model using numpy for speed.
    v11: No term limit, take LAST 50 subs (most recent/relevant).
    """
    batch_size = len(batch_data)

    # v11: Compute max_terms dynamically (no limit), keep max_subs=50 (model constraint)
    max_terms = max(len(filter_to_sector(d[0], d[3])) for d in batch_data) if batch_data else 1
    max_terms = max(max_terms, 1)  # Ensure at least 1
    max_subs = 50  # Model has fixed positional encoding for 50 subs
    max_actions = max(len(d[2]) for d in batch_data)
    max_actions = min(max_actions, 900)

    # Allocate numpy arrays
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

    n_valid_actions = []

    for i, (expr, subs, valid_actions, target_sector, target) in enumerate(batch_data):
        sector_expr = filter_to_sector(expr, target_sector)

        # v11: No limit on expr_items
        expr_items = list(sector_expr.items())
        n_expr = len(expr_items)
        if n_expr > 0:
            expr_integrals_np[i, :n_expr] = [integral for integral, _ in expr_items]
            expr_coeffs_np[i, :n_expr] = [coeff for _, coeff in expr_items]
            expr_mask_np[i, :n_expr] = True

        # v11: Take LAST 50 subs (most recent, for lower-weight integrals currently being reduced)
        all_subs = list(subs.items())
        if i == 0 and len(all_subs) > max_subs:
            print(f"  [DEBUG] subs count: {len(all_subs)}, taking last {max_subs}", flush=True)
        subs_items = all_subs[-max_subs:]
        for j, (key_integral, replacement) in enumerate(subs_items):
            sub_mask_np[i, j] = True
            sub_keys_np[i, j] = key_integral
            repl_items = list(replacement.items())[:MAX_REPLACEMENT_TERMS]
            n_repl = len(repl_items)
            if n_repl > 0:
                sub_repl_ints_np[i, j, :n_repl] = [ri for ri, _ in repl_items]
                sub_repl_coeffs_np[i, j, :n_repl] = [c for _, c in repl_items]
                sub_repl_mask_np[i, j, :n_repl] = True

        n_actions = min(len(valid_actions), max_actions)
        n_valid_actions.append(n_actions)
        if n_actions > 0:
            action_ibp_ops_np[i, :n_actions] = [op for op, _ in valid_actions[:n_actions]]
            action_deltas_np[i, :n_actions] = [delta for _, delta in valid_actions[:n_actions]]
            action_mask_np[i, :n_actions] = True

        sector_masks_np[i] = get_sector_mask(target)
        target_integrals_np[i] = target

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
                target_sector=None, filter_mode='subsector', track_integrals=None,
                use_resolved_subs=True, n_workers=16, prime=1009, patience=None,
                stop_on_weight_improvement=False):
    """Beam search with state-level parallelization.

    Args:
        n_workers: Number of worker processes for parallel state processing.
        prime: Prime for modular arithmetic (passed to workers).
        patience: If set, stop after this many steps without weight improvement.
                  If None, uses max_steps as hard limit.
        stop_on_weight_improvement: If True, stop as soon as max weight improves.
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
        resolved_subs={},
        score=0.0,
        path=[],
        n_non_masters=len(initial_non_masters)
    )

    beam = [initial_state]
    best_solution = None
    best_weight_ever = max_weight(start_expr, target_sector)
    initial_weight = best_weight_ever
    steps_since_improvement = 0

    # Initialize worker pool
    pool = None
    if n_workers > 1 and use_resolved_subs:
        pool = mp.Pool(processes=n_workers, initializer=init_worker, initargs=(prime,))
        print(f"  Initialized worker pool with {n_workers} workers", flush=True)

    try:
        step = 0
        while True:
            # Check stopping conditions
            if patience is not None:
                if steps_since_improvement >= patience:
                    if verbose:
                        print(f'Step {step}: STOPPING - no improvement in {patience} steps', flush=True)
                    break
            else:
                if step >= max_steps:
                    break

            step_start = time.time()
            if not beam:
                break

            # Check for success and track best weight
            weight_improved = False
            for state in beam:
                w = max_weight(state.expr, target_sector)
                if w < best_weight_ever:
                    best_weight_ever = w
                    weight_improved = True
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

            # Update patience counter
            if weight_improved:
                steps_since_improvement = 0
                # Early stop if requested
                if stop_on_weight_improvement:
                    if verbose:
                        print(f'Step {step}: STOPPING - weight improved (stop_on_weight_improvement=True)', flush=True)
                    break
            else:
                steps_since_improvement += 1

            if best_solution:
                break

            # PHASE 1: Collect all (state, target) pairs and get valid actions
            t1 = time.time()

            # First pass: collect all (state_idx, state, target) tuples
            tasks = []
            for state_idx, state in enumerate(beam):
                non_masters = get_non_masters(state.expr, target_sector)
                if not non_masters:
                    continue

                max_w = max((weight(k)[0], weight(k)[1]) for k in non_masters.keys())
                tied_targets = [k for k in non_masters.keys() if (weight(k)[0], weight(k)[1]) == max_w]

                for target in tied_targets:
                    tasks.append((state_idx, state, target, len(tied_targets)))

            # Get valid actions with parallel state processing
            if use_resolved_subs and pool is not None:
                valid_actions_list = [None] * len(tasks)

                # Group tasks by state
                from collections import defaultdict
                tasks_by_state = defaultdict(list)
                for task_idx, t in enumerate(tasks):
                    state_idx = t[0]
                    tasks_by_state[state_idx].append((task_idx, t))

                n_unique_states = len(tasks_by_state)

                # Prepare arguments for parallel processing
                worker_args = []
                for state_idx, state_tasks in tasks_by_state.items():
                    state = state_tasks[0][1][1]  # Get state from first task
                    targets_info = [(task_idx, t[2]) for task_idx, t in state_tasks]
                    worker_args.append((
                        state_idx,
                        state.subs,
                        state.resolved_subs,
                        targets_info,
                        target_sector,
                        filter_mode
                    ))

                # Parallel execution
                t_parallel = time.time()
                all_results = pool.map(process_state_tasks, worker_args)
                t_parallel_elapsed = time.time() - t_parallel

                # Collect results
                for state_results in all_results:
                    for task_idx, valid_actions in state_results:
                        valid_actions_list[task_idx] = valid_actions

                if verbose and step % 10 == 0:
                    print(f"    [P1 parallel] {n_unique_states} states across {n_workers} workers in {t_parallel_elapsed:.3f}s", flush=True)

            elif use_resolved_subs:
                # Sequential fallback (single worker or n_workers=1)
                valid_actions_list = [None] * len(tasks)
                from collections import defaultdict
                tasks_by_state = defaultdict(list)
                for task_idx, t in enumerate(tasks):
                    state_idx = t[0]
                    tasks_by_state[state_idx].append((task_idx, t))

                for state_idx, state_tasks in tasks_by_state.items():
                    state = state_tasks[0][1][1]

                    if target_sector is not None:
                        filtered_subs = filter_subs_to_exact_sector(state.subs, target_sector)
                        filtered_resolved = filter_resolved_subs_to_exact_sector(state.resolved_subs, target_sector)
                    else:
                        filtered_subs = state.subs
                        filtered_resolved = state.resolved_subs

                    indirect_cache = env.compute_indirect_cache(filtered_subs, filtered_resolved)

                    for task_idx, t in state_tasks:
                        target = t[2]
                        valid_actions_list[task_idx] = env.get_valid_actions_with_cache(
                            target, indirect_cache, filtered_subs, filtered_resolved,
                            filter_mode=filter_mode, verbose_timing=False
                        )
            else:
                valid_actions_list = []
                for t in tasks:
                    state = t[1]
                    target = t[2]
                    if target_sector is not None:
                        filtered_subs = filter_subs_to_exact_sector(state.subs, target_sector)
                    else:
                        filtered_subs = state.subs
                    valid_actions_list.append(
                        env.get_valid_actions_cached(target, filtered_subs, filter_mode=filter_mode)
                    )

            # Build batch_data and batch_meta from results
            batch_data = []
            batch_meta = []

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

            # PHASE 3: Extract top-k actions and create candidates (PARALLEL)
            t3 = time.time()
            candidates = []

            if use_resolved_subs and pool is not None:
                # Collect all actions to apply
                all_actions = []
                for i, (state_idx, target, valid_actions, n_tied_targets) in enumerate(batch_meta):
                    state = beam[state_idx]
                    n_valid = n_valid_actions[i]

                    k_per_target = max(1, beam_width // n_tied_targets)
                    top_k = min(k_per_target, n_valid)
                    top_indices = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()

                    for idx in top_indices:
                        ibp_op, delta = valid_actions[idx]
                        action_prob = probs[i, idx].item()
                        all_actions.append((
                            state.expr, state.subs, state.resolved_subs,
                            target, ibp_op, delta, action_prob,
                            state.path, state.score
                        ))

                # Distribute actions across workers
                if all_actions:
                    n_actions = len(all_actions)
                    chunk_size = max(1, n_actions // n_workers)
                    action_chunks = []
                    for j in range(0, n_actions, chunk_size):
                        chunk = all_actions[j:j + chunk_size]
                        action_chunks.append((chunk, target_sector))

                    # Parallel execution
                    t_p3_parallel = time.time()
                    all_results = pool.map(apply_actions_worker, action_chunks)
                    t_p3_parallel_elapsed = time.time() - t_p3_parallel

                    # Collect results
                    for chunk_results in all_results:
                        for result in chunk_results:
                            if result is not None:
                                new_expr, new_subs, new_resolved_subs, new_path, new_score, n_non_masters = result
                                candidates.append(State(
                                    expr=new_expr,
                                    subs=new_subs,
                                    resolved_subs=new_resolved_subs,
                                    score=new_score,
                                    path=new_path,
                                    n_non_masters=n_non_masters
                                ))

                    if verbose and step % 10 == 0:
                        print(f"    [P3 parallel] {n_actions} actions across {len(action_chunks)} chunks in {t_p3_parallel_elapsed:.3f}s", flush=True)
            else:
                # Sequential fallback
                for i, (state_idx, target, valid_actions, n_tied_targets) in enumerate(batch_meta):
                    state = beam[state_idx]
                    n_valid = n_valid_actions[i]

                    k_per_target = max(1, beam_width // n_tied_targets)
                    top_k = min(k_per_target, n_valid)
                    top_indices = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()

                    for idx in top_indices:
                        ibp_op, delta = valid_actions[idx]
                        action_prob = probs[i, idx].item()

                        if use_resolved_subs:
                            new_expr, new_subs, new_resolved_subs, success = env.apply_action_resolved(
                                dict(state.expr), dict(state.subs), state.resolved_subs, target, ibp_op, delta
                            )
                        else:
                            new_expr, new_subs, success = env.apply_action(
                                dict(state.expr), dict(state.subs), target, ibp_op, delta
                            )
                            new_resolved_subs = {}

                        if not success:
                            continue

                        new_non_masters = get_non_masters(new_expr, target_sector)
                        new_path = state.path + [(target, ibp_op, delta)]
                        new_score = state.score + torch.log(torch.tensor(action_prob + 1e-10)).item()

                        candidates.append(State(
                            expr=new_expr,
                            subs=new_subs,
                            resolved_subs=new_resolved_subs,
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

            step += 1

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return best_solution, beam, best_weight_ever


def main():
    parser = argparse.ArgumentParser(description='Beam search reduction with classifier v5 (parallel)')
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
                        help='Sector filtering mode')
    parser.add_argument('--n_workers', type=int, default=16,
                        help='Number of worker processes for parallel state processing')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('Beam Search Reduction with Classifier v5 (Parallel)', flush=True)
    print('  - State-level parallelization', flush=True)
    print(f'  - {args.n_workers} worker processes', flush=True)
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
        filter_mode=args.filter_mode,
        n_workers=args.n_workers,
        prime=args.prime
    )

    print('=' * 70, flush=True)
    print('\nResults:', flush=True)
    print(f'Best weight achieved: {best_weight}', flush=True)

    if solution:
        print(f'\nSUCCESS! Found complete reduction in {len(solution.path)} steps', flush=True)
        print('\nReduction path:', flush=True)
        for i, (target, ibp_op, delta) in enumerate(solution.path):
            print(f'  {i+1}. Eliminate I{list(target)} using IBP_{ibp_op}, delta={list(delta)}', flush=True)

        # Print final expression
        print('\n' + '=' * 70, flush=True)
        print('FULL FINAL EXPRESSION (ALL INTEGRALS):', flush=True)
        print('=' * 70, flush=True)
        final_expr = solution.expr
        print(f'Total integrals in expression: {len(final_expr)}', flush=True)

        sectors = {}
        for integral, coeff in final_expr.items():
            sector = tuple(get_sector_mask(integral))
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append((integral, coeff))

        print(f'Number of distinct sectors: {len(sectors)}', flush=True)

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
