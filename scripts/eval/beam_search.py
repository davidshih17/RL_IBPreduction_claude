#!/usr/bin/env python3
"""
Beam search reduction using classifier v5 with full parallelization.

v13 changes from v11:
- P2 sub-batching: process model inference in chunks of 50 states to limit
  peak memory from large batch tensors in prepare_batched_input_v5

v11 changes from v7:
- Remove max_terms=200 limit (model sees full expression)
- Take LAST 50 subs instead of first 50 (most recent = most relevant for current weight)

Usage:
    python -u scripts/eval/beam_search.py --beam_width 10 --max_steps 100 --n_workers 16
"""

import sys
import math
import argparse
from pathlib import Path
from collections import namedtuple
import multiprocessing as mp
from functools import partial

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import numpy as np
import torch
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, filter_top_sector, is_master, weight, resolve_subs,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector
)
from sailir.classifier import IBPActionClassifier
from sailir.topology import Topology

# Reuse core functions from v4
from beam_search_utils import (
    get_sector_mask,
    filter_to_sector,
    get_non_masters,
    max_weight
)

# Our own State that includes resolved_subs for optimization.
# indirect_aux: optional 4-tuple (cached_unique, union_bms, raw_id_to_idx,
# indirect_raws) produced by compute_indirect_substituted_with_aux. When set,
# process_state_tasks can rebuild indirect_cache from aux without re-running
# the expensive apply_resolved_subs_batch.
# sub_accum: dict of sub-sector integrals → coefficients. These are "passenger"
# terms that accumulate via substitutions but never affect target selection,
# action enumeration, or model decisions (those all filter to target sector).
# Keeping them OUT of `expr` saves O(|sub_accum|) per apply_substitution call
# in apply_actions_worker (~5x speedup at late stages). Combined back into the
# full expr only when the worker returns its final result to the orchestrator.
State = namedtuple('State',
                   ['expr', 'subs', 'resolved_subs', 'score', 'path', 'n_non_masters',
                    'indirect_aux', 'sub_accum'])
State.__new__.__defaults__ = (None, None)  # both default to None

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
        Tuple (results_list, state_idx, profile_data_or_None). profile_data is
        a dict with sizes + timings if env var BEAM_PROFILE_CSV is set, else None.
    """
    global _worker_env
    import os, time

    state_idx, state_subs, state_resolved_subs, targets_info, target_sector, filter_mode = args
    profile = bool(os.environ.get("BEAM_PROFILE_CSV"))
    profile_data = None
    if profile:
        profile_data = {
            'state_idx': state_idx,
            'len_subs': len(state_subs),
            'len_resolved': len(state_resolved_subs),
            'sum_resolved_value_sizes': sum(len(v) for v in state_resolved_subs.values()),
            'n_targets': len(targets_info),
        }

    # Filter subs for this state
    t0 = time.time() if profile else 0
    if target_sector is not None:
        filtered_subs = filter_subs_to_exact_sector(state_subs, target_sector)
        filtered_resolved = filter_resolved_subs_to_exact_sector(state_resolved_subs, target_sector)
    else:
        filtered_subs = state_subs
        filtered_resolved = state_resolved_subs
    if profile:
        profile_data['t_filter'] = time.time() - t0
        profile_data['len_filtered_subs'] = len(filtered_subs)
        profile_data['len_filtered_resolved'] = len(filtered_resolved)
        profile_data['sum_filtered_resolved_value_sizes'] = sum(
            len(v) for v in filtered_resolved.values())

    # Compute indirect cache ONCE for this state
    t0 = time.time() if profile else 0
    indirect_cache = _worker_env.compute_indirect_cache(filtered_subs, filtered_resolved)
    if profile:
        profile_data['t_compute_indirect_cache'] = time.time() - t0
        profile_data['len_indirect_cache'] = len(indirect_cache)

    # Get valid actions for each target
    results = []
    t_get_valid_total = 0.0
    if profile:
        # Clear any prior accumulator in this worker process so we only
        # capture per-state get-valid profile rows.
        from sailir import ibp_env as _ibp_env_mod
        _ibp_env_mod._GETVALID_PROFILE = []
    for task_idx, target in targets_info:
        t0 = time.time() if profile else 0
        valid_actions = _worker_env.get_valid_actions_with_cache(
            target, indirect_cache, filtered_subs, filtered_resolved,
            filter_mode=filter_mode, verbose_timing=False
        )
        if profile:
            t_get_valid_total += time.time() - t0
        results.append((task_idx, valid_actions))

    if profile:
        profile_data['t_get_valid_total'] = t_get_valid_total
        # Aggregate get-valid sub-counters across all targets in this state.
        agg = {
            'n_calls': 0, 'n_indirect_checked': 0,
            'n_pass_sc1': 0, 'n_pass_sc2': 0, 'n_pass_filter': 0,
            't_sc1': 0.0, 't_sc2': 0.0, 't_filter_in_loop': 0.0,
            't_p1a_direct': 0.0, 't_p1b_indirect': 0.0,
            'sum_cached_size_at_sc2': 0, 'sum_cached_size_at_filter': 0,
        }
        for row in _ibp_env_mod._GETVALID_PROFILE:
            agg['n_calls'] += 1
            for k in agg:
                if k == 'n_calls':
                    continue
                agg[k] += row[k]
        profile_data['getvalid'] = agg

    return (results, state_idx, profile_data)


def apply_actions_worker(args):
    """Worker function: apply a batch of actions and return candidate states.

    Args:
        args: tuple of (action_batch, target_sector)
              action_batch is list of (expr_t, subs, resolved_subs,
                                       target, ibp_op, delta, action_prob,
                                       path, score)
              expr_t = target-sector expression dict ONLY. Sub-sector content
              is dropped during beam_search (Option F) and reconstructed at
              the worker's end by replaying path+subs against start_expr.

    Returns:
        List of (new_expr_t, new_subs, new_resolved_subs, new_path,
                 new_score, n_non_masters) or None if failed.
    """
    global _worker_env

    action_batch, target_sector = args
    results = []

    for expr_t, subs, resolved_subs, target, ibp_op, delta, action_prob, path, score in action_batch:
        # Pass expr_t / subs directly (no dict() wrapper). The callee makes its
        # own copies internally (new_subs = dict(subs)), and uses
        # apply_substitution_target_only / add_sub_to_resolved which both return
        # fresh dicts. So the outer dict() wraps were ~60 µs of pure waste per
        # action at |subs|=300 (microbenchmark: 75% "residual" overhead).
        new_expr_t, new_subs, new_resolved_subs, success = (
            _worker_env.apply_action_resolved_target_only(
                expr_t, subs, resolved_subs,
                target, ibp_op, delta, target_sector,
            )
        )

        if not success:
            results.append(None)
            continue

        # n_non_masters = non-master keys in target-sector expr_t (which is
        # exactly the target-sector content by construction).
        n_non_masters = sum(1 for integral in new_expr_t.keys() if not is_master(integral))

        new_path = path + [(target, ibp_op, delta)]
        new_score = score + math.log(action_prob + 1e-10)

        results.append((new_expr_t, new_subs, new_resolved_subs,
                        new_path, new_score, n_non_masters))

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

    N = ibp_env.N_INDICES
    D = ibp_env.N_DENOMINATORS

    # Allocate numpy arrays
    expr_integrals_np = np.zeros((batch_size, max_terms, N), dtype=np.int64)
    expr_coeffs_np = np.zeros((batch_size, max_terms), dtype=np.int64)
    expr_mask_np = np.zeros((batch_size, max_terms), dtype=np.bool_)

    sub_keys_np = np.zeros((batch_size, max_subs, N), dtype=np.int64)
    sub_repl_ints_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS, N), dtype=np.int64)
    sub_repl_coeffs_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.int64)
    sub_repl_mask_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.bool_)
    sub_mask_np = np.zeros((batch_size, max_subs), dtype=np.bool_)

    action_ibp_ops_np = np.zeros((batch_size, max_actions), dtype=np.int64)
    action_deltas_np = np.zeros((batch_size, max_actions, N), dtype=np.int64)
    action_mask_np = np.zeros((batch_size, max_actions), dtype=np.bool_)

    sector_masks_np = np.zeros((batch_size, D), dtype=np.int64)
    target_integrals_np = np.zeros((batch_size, N), dtype=np.int64)

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
                stop_on_weight_improvement=False, random_target=False, beam_sort='mixed',
                checkpoint_path=None, checkpoint_interval=50, checkpoint_time_seconds=300,
                resume_from=None, dedup_beam_by_content=False):
    """Beam search with state-level parallelization.

    Args:
        n_workers: Number of worker processes for parallel state processing.
        prime: Prime for modular arithmetic (passed to workers).
        patience: If set, stop after this many steps without weight improvement.
                  If None, uses max_steps as hard limit.
        stop_on_weight_improvement: If True, stop as soon as max weight improves.
        random_target: If True, pick one random tied max-weight target per state
                       instead of trying all tied targets (adds stochasticity).
        beam_sort: Sort key for beam candidates. Options:
            'weight': (max_weight, n_non_masters, -score) - greedy weight reduction [default]
            'nterms': (n_non_masters, max_weight, -score) - prioritize fewer non-masters
            'score': (-score, max_weight, n_non_masters) - trust model confidence
            'totalweight': (sum_weight, n_non_masters, -score) - reduce total complexity
            'mixed': two parallel beams of beam_width each (weight + totalweight)
    """
    import time
    import random

    def _total_weight(expr, ts):
        """Sum of (w0, w1) over all non-masters."""
        nms = get_non_masters(expr, ts)
        if not nms:
            return (0, 0)
        return (sum(weight(k)[0] for k in nms), sum(weight(k)[1] for k in nms))

    def _sort_key_weight(s):
        mw = max_weight(s.expr, target_sector)
        return (mw, s.n_non_masters, -s.score)

    def _sort_key_totalweight(s):
        tw = _total_weight(s.expr, target_sector)
        return (tw, s.n_non_masters, -s.score)

    def _sort_key(s):
        mw = max_weight(s.expr, target_sector)
        if beam_sort == 'weight':
            return (mw, s.n_non_masters, -s.score)
        elif beam_sort == 'nterms':
            return (s.n_non_masters, mw, -s.score)
        elif beam_sort == 'score':
            return (-s.score, mw, s.n_non_masters)
        elif beam_sort == 'totalweight':
            tw = _total_weight(s.expr, target_sector)
            return (tw, s.n_non_masters, -s.score)
        elif beam_sort == 'mixed':
            pass  # handled separately in beam selection
        elif beam_sort == 'mixed_dedup':
            pass  # handled separately: half dedup, half no-dedup
        else:
            raise ValueError(f"Unknown beam_sort: {beam_sort}")

    def _select_mixed_beam(candidates, beam_width):
        """Select beam by taking top half from weight sort and top half from totalweight sort."""
        half = beam_width // 2
        # Sort by weight, take top half
        by_weight = sorted(candidates, key=_sort_key_weight)
        # Sort by totalweight, take top half
        by_total = sorted(candidates, key=_sort_key_totalweight)

        # Build beam: weight picks first, then fill with totalweight picks (no duplicates)
        beam = []
        seen = set()
        for s in by_weight:
            if len(beam) >= half:
                break
            sid = id(s)
            if sid not in seen:
                seen.add(sid)
                beam.append(s)
        for s in by_total:
            if len(beam) >= beam_width:
                break
            sid = id(s)
            if sid not in seen:
                seen.add(sid)
                beam.append(s)
        # If still short (due to overlap), fill from weight
        for s in by_weight:
            if len(beam) >= beam_width:
                break
            sid = id(s)
            if sid not in seen:
                seen.add(sid)
                beam.append(s)
        return beam

    # Initialize tracking
    if track_integrals:
        tracked_coeffs = {i: start_expr.get(i, 0) for i in track_integrals}
        print(f"  [TRACK] Initial coefficients:", flush=True)
        for i, c in tracked_coeffs.items():
            print(f"    I{list(i)} = {c}", flush=True)

    initial_non_masters = get_non_masters(start_expr, target_sector)
    # Split start_expr into target-sector and sub-sector parts.
    # The split state.expr only ever carries target-sector content; sub-sector
    # terms live in state.sub_accum as passengers (never read by next-step
    # decisions; only re-attached when worker returns final result).
    if target_sector is not None:
        _init_expr_t = {k: v for k, v in start_expr.items()
                        if tuple(1 if k[i] > 0 else 0
                                  for i in range(ibp_env.N_DENOMINATORS)) == target_sector}
        _init_sub_accum = {k: v for k, v in start_expr.items()
                           if k not in _init_expr_t}
    else:
        _init_expr_t = dict(start_expr)
        _init_sub_accum = {}
    initial_state = State(
        expr=_init_expr_t,
        subs={},
        resolved_subs={},
        score=0.0,
        path=[],
        n_non_masters=len(initial_non_masters),
        indirect_aux=None,
        sub_accum=_init_sub_accum,
    )

    # Tabu list of expr-hashes (target-sector exprs) that have appeared in the
    # beam at any previous step. Used by the dedup to prevent the search from
    # revisiting the same expr-region — which empirically formed long cycles
    # (e.g. expr 8f3116afda revisited 20+ times across 200 steps under plain
    # dedup-by-expr). This loses the rs-diversity that dedup-OFF preserves,
    # but the user judged that contribution minor compared to breaking cycles.
    visited_exprs = set()
    visited_exprs.add(frozenset(initial_state.expr.items()))

    beam = [initial_state]
    weight_beam_end = len(beam)  # For mixed mode: boundary between weight/total halves
    best_solution = None
    best_weight_ever = max_weight(start_expr, target_sector)
    initial_weight = best_weight_ever
    steps_since_improvement = 0
    resumed_step = 0  # set non-zero if we loaded from a checkpoint

    # Resume from checkpoint if requested. Checkpoint format:
    #   {step, beam, best_weight_ever, best_solution, steps_since_improvement,
    #    weight_beam_end, initial_weight}
    # written atomically by the same beam_search loop below.
    if resume_from is not None:
        import pickle
        from pathlib import Path
        cp = Path(resume_from)
        if cp.exists():
            with open(cp, 'rb') as f:
                ck = pickle.load(f)
            resumed_step = int(ck['step'])
            beam_raw = list(ck['beam'])
            best_weight_ever = ck['best_weight_ever']
            best_solution = ck.get('best_solution')
            steps_since_improvement = int(ck.get('steps_since_improvement', 0))
            weight_beam_end = int(ck.get('weight_beam_end', len(beam_raw)))
            initial_weight = ck.get('initial_weight', initial_weight)
            # Migrate old beam states (6 or 7 fields) to new shape (8 fields).
            # Specifically: if a state lacks sub_accum, split its expr now.
            beam = []
            for s in beam_raw:
                if hasattr(s, 'sub_accum') and s.sub_accum is not None:
                    beam.append(s)
                else:
                    # Split: target-sector content stays in expr; rest is sub_accum.
                    if target_sector is not None:
                        et = {k: v for k, v in s.expr.items()
                              if tuple(1 if k[i] > 0 else 0
                                       for i in range(ibp_env.N_DENOMINATORS)) == target_sector}
                        sa = {k: v for k, v in s.expr.items() if k not in et}
                    else:
                        et = dict(s.expr); sa = {}
                    # Reconstruct State with proper fields (handle both old shapes).
                    n_nm = getattr(s, 'n_non_masters',
                                   sum(1 for k in et if not is_master(k)))
                    beam.append(State(
                        expr=et, subs=s.subs, resolved_subs=s.resolved_subs,
                        score=s.score, path=s.path, n_non_masters=n_nm,
                        indirect_aux=getattr(s, 'indirect_aux', None),
                        sub_accum=sa,
                    ))
            if verbose:
                print(f"  Resumed from {resume_from} at step {resumed_step}, "
                      f"beam={len(beam)}, best_weight_ever={best_weight_ever}", flush=True)
        else:
            print(f"  WARNING: resume_from={resume_from} does not exist; starting fresh.", flush=True)

    # Initialize worker pool
    pool = None
    if n_workers > 1 and use_resolved_subs:
        pool = mp.Pool(processes=n_workers, initializer=init_worker, initargs=(prime,))
        print(f"  Initialized worker pool with {n_workers} workers", flush=True)

    def _save_checkpoint(step_now):
        """Atomically dump current beam_search state to checkpoint_path."""
        if not checkpoint_path:
            return
        import pickle
        from pathlib import Path
        cp = Path(checkpoint_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(cp.suffix + '.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump({
                'step': step_now,
                'beam': beam,
                'best_weight_ever': best_weight_ever,
                'best_solution': best_solution,
                'steps_since_improvement': steps_since_improvement,
                'weight_beam_end': weight_beam_end,
                'initial_weight': initial_weight,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cp)
        if verbose:
            print(f"  Checkpoint saved at step {step_now} -> {cp}", flush=True)

    last_checkpoint_time = time.time()

    try:
        step = resumed_step
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

                if random_target:
                    target = random.choice(tied_targets)
                    tasks.append((state_idx, state, target, 1))
                else:
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
                import os
                _profile_csv = os.environ.get("BEAM_PROFILE_CSV")
                for entry in all_results:
                    state_results, _state_idx, profile_data = entry
                    for task_idx, valid_actions in state_results:
                        valid_actions_list[task_idx] = valid_actions
                    # Optional per-state profile row.
                    if _profile_csv and profile_data is not None:
                        new_file = not os.path.exists(_profile_csv)
                        gv = profile_data.get('getvalid') or {}
                        with open(_profile_csv, 'a') as _f:
                            cols = ['step', 'state_idx', 'len_subs', 'len_resolved',
                                    'sum_resolved_value_sizes', 'len_filtered_subs',
                                    'len_filtered_resolved', 'sum_filtered_resolved_value_sizes',
                                    'len_indirect_cache', 'n_targets', 't_filter',
                                    't_compute_indirect_cache', 't_get_valid_total',
                                    # Get-valid drill-down counters (summed over targets in this state)
                                    'gv_n_calls', 'gv_n_indirect_checked',
                                    'gv_n_pass_sc1', 'gv_n_pass_sc2', 'gv_n_pass_filter',
                                    'gv_t_sc1', 'gv_t_sc2', 'gv_t_filter_in_loop',
                                    'gv_t_p1a_direct', 'gv_t_p1b_indirect',
                                    'gv_sum_cached_size_at_sc2', 'gv_sum_cached_size_at_filter']
                            if new_file:
                                _f.write(','.join(cols) + '\n')
                            row = [str(step), str(profile_data['state_idx']),
                                   str(profile_data['len_subs']),
                                   str(profile_data['len_resolved']),
                                   str(profile_data['sum_resolved_value_sizes']),
                                   str(profile_data['len_filtered_subs']),
                                   str(profile_data['len_filtered_resolved']),
                                   str(profile_data['sum_filtered_resolved_value_sizes']),
                                   str(profile_data['len_indirect_cache']),
                                   str(profile_data['n_targets']),
                                   f"{profile_data['t_filter']:.4f}",
                                   f"{profile_data['t_compute_indirect_cache']:.4f}",
                                   f"{profile_data['t_get_valid_total']:.4f}",
                                   str(gv.get('n_calls', 0)),
                                   str(gv.get('n_indirect_checked', 0)),
                                   str(gv.get('n_pass_sc1', 0)),
                                   str(gv.get('n_pass_sc2', 0)),
                                   str(gv.get('n_pass_filter', 0)),
                                   f"{gv.get('t_sc1', 0.0):.4f}",
                                   f"{gv.get('t_sc2', 0.0):.4f}",
                                   f"{gv.get('t_filter_in_loop', 0.0):.4f}",
                                   f"{gv.get('t_p1a_direct', 0.0):.4f}",
                                   f"{gv.get('t_p1b_indirect', 0.0):.4f}",
                                   str(gv.get('sum_cached_size_at_sc2', 0)),
                                   str(gv.get('sum_cached_size_at_filter', 0))]
                            _f.write(','.join(row) + '\n')

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

            # PHASE 2: Batched model inference (sub-batched to limit memory)
            t2 = time.time()
            P2_SUB_BATCH = 50  # Process at most 50 states at a time through model

            if len(batch_data) <= P2_SUB_BATCH:
                # Small enough to do in one shot
                batch, n_valid_actions = prepare_batched_input_v5(batch_data, device)
                with torch.no_grad():
                    logits, probs = model(
                        batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
                        batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
                        batch['sub_repl_mask'], batch['sub_mask'],
                        batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
                        batch['sector_mask'], batch['target_integral']
                    )
                del batch
            else:
                # Sub-batch to limit peak memory from model intermediates
                # Store per-item logits/probs since sub-batches have different max_actions dims
                all_logits_list = []  # list of 1D tensors, one per item
                all_probs_list = []
                n_valid_actions = []
                for sb_start in range(0, len(batch_data), P2_SUB_BATCH):
                    sb_data = batch_data[sb_start:sb_start + P2_SUB_BATCH]
                    sb_batch, sb_nva = prepare_batched_input_v5(sb_data, device)
                    with torch.no_grad():
                        sb_logits, sb_probs = model(
                            sb_batch['expr_integrals'], sb_batch['expr_coeffs'], sb_batch['expr_mask'],
                            sb_batch['sub_keys'], sb_batch['sub_repl_ints'], sb_batch['sub_repl_coeffs'],
                            sb_batch['sub_repl_mask'], sb_batch['sub_mask'],
                            sb_batch['action_ibp_ops'], sb_batch['action_deltas'], sb_batch['action_mask'],
                            sb_batch['sector_mask'], sb_batch['target_integral']
                        )
                    # Extract per-item results before deleting batch tensors
                    for j in range(len(sb_data)):
                        nv = sb_nva[j]
                        all_logits_list.append(sb_logits[j, :nv].cpu())
                        all_probs_list.append(sb_probs[j, :nv].cpu())
                    n_valid_actions.extend(sb_nva)
                    del sb_batch, sb_logits, sb_probs
                # Pad and stack into uniform tensors for Phase 3
                max_nv = max(n_valid_actions)
                logits = torch.zeros(len(batch_data), max_nv, device=device)
                probs = torch.zeros(len(batch_data), max_nv, device=device)
                for j in range(len(batch_data)):
                    nv = n_valid_actions[j]
                    logits[j, :nv] = all_logits_list[j]
                    probs[j, :nv] = all_probs_list[j]
                del all_logits_list, all_probs_list

            t2_elapsed = time.time() - t2

            # PHASE 3: Extract top-k actions and create candidates (PARALLEL)
            t3 = time.time()
            candidates = []
            candidate_halves = []  # For mixed mode: 0=weight beam, 1=total beam
            # candidate_parents: id(state)->parent state_idx (in PREVIOUS beam).
            # Used by _take_top_unique_by_content when MAX_CHILDREN_PER_PARENT
            # env var is set, to limit how many children any single parent can
            # contribute to the next beam (beam diversity preservation).
            candidate_parents = {}

            if use_resolved_subs and pool is not None:
                # Collect all actions to apply
                all_actions = []
                all_action_halves = []  # For mixed mode: track source beam
                all_action_parents = []  # parent state_idx parallel to all_actions
                for i, (state_idx, target, valid_actions, n_tied_targets) in enumerate(batch_meta):
                    state = beam[state_idx]
                    n_valid = n_valid_actions[i]

                    k_per_target = max(1, beam_width // n_tied_targets)
                    top_k = min(k_per_target, n_valid)
                    top_indices = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()

                    # action_half: 0 if state is in first half of beam (dedup or
                    # weight-sorted), 1 if in second half (no-dedup or
                    # totalweight-sorted). Only meaningful in 'mixed' /
                    # 'mixed_dedup' modes; otherwise everything is half 0.
                    if beam_sort in ('mixed', 'mixed_dedup'):
                        action_half = 0 if state_idx < weight_beam_end else 1
                    else:
                        action_half = 0
                    for idx in top_indices:
                        ibp_op, delta = valid_actions[idx]
                        action_prob = probs[i, idx].item()
                        all_actions.append((
                            state.expr, state.subs, state.resolved_subs,
                            target, ibp_op, delta, action_prob,
                            state.path, state.score
                        ))
                        all_action_halves.append(action_half)
                        all_action_parents.append(state_idx)

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
                    flat_idx = 0
                    for chunk_results in all_results:
                        for result in chunk_results:
                            if result is not None:
                                new_expr_t, new_subs, new_resolved_subs, new_path, new_score, n_non_masters = result
                                _cand = State(
                                    expr=new_expr_t,
                                    subs=new_subs,
                                    resolved_subs=new_resolved_subs,
                                    score=new_score,
                                    path=new_path,
                                    n_non_masters=n_non_masters,
                                    indirect_aux=None,
                                    sub_accum=None,  # Option F: no sub_accum tracking; reconstruct at end
                                )
                                candidates.append(_cand)
                                candidate_halves.append(all_action_halves[flat_idx])
                                candidate_parents[id(_cand)] = all_action_parents[flat_idx]
                            flat_idx += 1

                    if verbose and step % 10 == 0:
                        print(f"    [P3 parallel] {n_actions} actions across {len(action_chunks)} chunks in {t_p3_parallel_elapsed:.3f}s", flush=True)
            else:
                # Sequential fallback with running buffer (v13)
                for i, (state_idx, target, valid_actions, n_tied_targets) in enumerate(batch_meta):
                    state = beam[state_idx]
                    n_valid = n_valid_actions[i]

                    k_per_target = max(1, beam_width // n_tied_targets)
                    top_k = min(k_per_target, n_valid)
                    top_indices = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()

                    action_half = 0 if (beam_sort != 'mixed' or state_idx < weight_beam_end) else 1
                    for idx in top_indices:
                        ibp_op, delta = valid_actions[idx]
                        action_prob = probs[i, idx].item()

                        # Option F: only target-sector content tracked during
                        # beam_search; sub-sector terms reconstructed at the
                        # end by replaying path through state.subs against the
                        # original start_expr (see onestep_worker._full_expr).
                        new_expr_t, new_subs, new_resolved_subs, success = (
                            env.apply_action_resolved_target_only(
                                state.expr, state.subs,
                                state.resolved_subs,
                                target, ibp_op, delta, target_sector,
                            )
                        )

                        if not success:
                            continue

                        n_nm = sum(1 for k in new_expr_t.keys() if not is_master(k))
                        new_path = state.path + [(target, ibp_op, delta)]
                        # NOTE: must match parallel apply_actions_worker exactly (math.log,
                        # float64) so 1-CPU and N-CPU runs follow identical beam trajectories.
                        new_score = state.score + math.log(action_prob + 1e-10)

                        _cand = State(
                            expr=new_expr_t,
                            subs=new_subs,
                            resolved_subs=new_resolved_subs,
                            score=new_score,
                            path=new_path,
                            n_non_masters=n_nm,
                            indirect_aux=None,
                            sub_accum=None,  # Option F: reconstructed at worker end
                        )
                        candidates.append(_cand)
                        candidate_halves.append(action_half)
                        candidate_parents[id(_cand)] = state_idx

            def _take_top_unique_by_content(sorted_cands, k):
                """Take up to k candidates from a sorted list, keeping only the
                first occurrence of each unique (target_sector_expr,
                resolved_subs) future-equivalence class.

                Identity for future evolution under Option F + storage stripping:
                  - target_sector_expr determines target selection and the
                    expr-side context the model sees.
                  - resolved_subs determines `cached = apply_resolved_subs(raw,
                    resolved_subs)` and thus the sol returned by solve_ibp_for
                    for any chosen action — i.e., the next step's expr.
                  - The model itself doesn't use subs values, so two states with
                    the same expr and same resolved_subs will produce the same
                    candidates with the same coefficients next step (truly
                    identical futures).

                Since resolved_subs values are now stripped to target-sector
                content at storage time, the full hash
                  frozenset((k, frozenset(v.items())) for k, v in resolved_subs.items())
                is cheap. Earlier dedup attempts (expr-only or resolved_subs-only)
                under-/over-merged because either side alone misses real
                future-diverging diversity.

                If fewer than k unique are available, pad with the best
                duplicates so the beam never shrinks.
                """
                import os as _os_dbg
                _dbg = bool(_os_dbg.environ.get('DEDUP_DEBUG'))
                _dump = _os_dbg.environ.get('DEDUP_DUMP_PAIR')
                # DEDUP_VARIANT controls the dedup key:
                #   'expr' (default): frozenset(expr.items()) — current
                #   'rs'            : frozenset(resolved_subs items, fully
                #                     canonicalized) — historical variant 2,
                #                     empirically cracked (8,4)
                #   'expr_rs'       : both — strictest, tested and worse
                _variant = _os_dbg.environ.get('DEDUP_VARIANT', 'expr')
                # NO_TABU=1 disables the visited_exprs tabu check below
                # (the tabu was added later to break expr-cycles seen with
                # the 'expr' variant; the original wide_beam_dedup_v1 had
                # no tabu, so to reproduce its behavior under Option F set
                # NO_TABU=1).
                _no_tabu = bool(_os_dbg.environ.get('NO_TABU'))
                # MAX_CHILDREN_PER_PARENT=N caps how many children any single
                # parent state (in the PREVIOUS beam) can contribute to the
                # NEXT beam. Encourages beam-state diversity at small beam
                # widths: prevents one strong parent from dominating all
                # slots. Unset / 0 → no cap. Uses the candidate_parents map
                # built in the outer scope of each step.
                _max_per_parent_str = _os_dbg.environ.get('MAX_CHILDREN_PER_PARENT')
                _max_per_parent = int(_max_per_parent_str) if _max_per_parent_str and int(_max_per_parent_str) > 0 else None
                _parent_count = {} if _max_per_parent is not None else None
                seen = set()
                seen_first = {}
                unique = []
                leftovers = []
                dup_samples = []
                _dumped = False
                # Dedup by chosen key AND exclude previously-visited exprs
                # (tabu list). Tabu always uses expr-key, regardless of dedup
                # variant — we don't want to revisit literal expr regions.
                for s in sorted_cands:
                    if _variant == 'rs':
                        key = frozenset(
                            (sk, frozenset(sv.items()))
                            for sk, sv in s.resolved_subs.items()
                        )
                    elif _variant == 'expr_rs':
                        rs_key = frozenset(
                            (sk, frozenset(sv.items()))
                            for sk, sv in s.resolved_subs.items()
                        )
                        key = (frozenset(s.expr.items()), rs_key)
                    else:  # 'expr' (default)
                        key = frozenset(s.expr.items())
                    expr_key_for_tabu = frozenset(s.expr.items())
                    is_tabu = (not _no_tabu) and (expr_key_for_tabu in visited_exprs)
                    # Parent-quota rejection: track parent of s (None if unknown,
                    # which only happens for resumed/migrated states from old
                    # checkpoints — in that case the quota does not apply to s).
                    _p = candidate_parents.get(id(s)) if _parent_count is not None else None
                    _over_quota = (_parent_count is not None and _p is not None
                                   and _parent_count.get(_p, 0) >= _max_per_parent)
                    if is_tabu or key in seen or _over_quota:
                        leftovers.append(s)
                        if _dbg and len(dup_samples) < 3:
                            dup_samples.append((len(s.expr), len(s.resolved_subs),
                                                s.score, s.n_non_masters))
                        # Dump the FIRST duplicate pair to a file for offline
                        # comparison (set DEDUP_DUMP_PAIR=<path>).
                        if _dump and not _dumped:
                            import pickle as _pkl
                            try:
                                with open(_dump, 'wb') as _f:
                                    _pkl.dump({
                                        'state_kept': seen_first[key],
                                        'state_skipped': s,
                                        'target_sector': target_sector,
                                    }, _f, protocol=_pkl.HIGHEST_PROTOCOL)
                                print(f"  [DEDUP] dumped colliding pair to {_dump}",
                                      flush=True)
                                _dumped = True
                            except Exception as _e:
                                print(f"  [DEDUP] dump failed: {_e}", flush=True)
                        continue
                    seen.add(key)
                    seen_first[key] = s
                    unique.append(s)
                    if _parent_count is not None and _p is not None:
                        _parent_count[_p] = _parent_count.get(_p, 0) + 1
                    if not _dbg and len(unique) >= k:
                        break
                if len(unique) < k:
                    unique.extend(leftovers[: k - len(unique)])
                if _dbg:
                    n_total = len(sorted_cands)
                    n_unique_keys = len(seen)
                    n_duplicate = n_total - n_unique_keys
                    print(
                        f"  [DEDUP] cands={n_total} unique_keys={n_unique_keys} "
                        f"duplicates={n_duplicate} kept_in_beam={len(unique[:k])} "
                        f"(of those {sum(1 for s in unique[:k] if s in unique)}/k unique)",
                        flush=True,
                    )
                    if dup_samples:
                        print(f"    dup_samples (|expr|, |rs|, score, NM): {dup_samples}",
                              flush=True)
                    unique = unique[:k]
                return unique

            # Select top beam_width candidates
            if beam_sort == 'mixed':
                # Parallel beams: each beam independently selects top beam_width
                w_cands = [c for c, h in zip(candidates, candidate_halves) if h == 0]
                t_cands = [c for c, h in zip(candidates, candidate_halves) if h == 1]
                # Bootstrap: if total beam was empty (first step), use all candidates
                if not t_cands:
                    t_cands = list(candidates)
                w_cands.sort(key=_sort_key_weight)
                t_cands.sort(key=_sort_key_totalweight)
                if dedup_beam_by_content:
                    w_selected = _take_top_unique_by_content(w_cands, beam_width)
                    t_selected = _take_top_unique_by_content(t_cands, beam_width)
                else:
                    w_selected = w_cands[:beam_width]
                    t_selected = t_cands[:beam_width]
                # Deduplicate total against weight (avoid processing same state twice)
                w_ids = set(id(s) for s in w_selected)
                t_deduped = [s for s in t_selected if id(s) not in w_ids]
                beam = w_selected + t_deduped
                weight_beam_end = len(w_selected)
            elif beam_sort == 'mixed_dedup':
                # Two parallel sub-beams of width beam_width//2 each:
                #   half A (expr-diversity): expr-dedup applied, forces 10
                #     distinct exprs in the beam
                #   half B (rs-diversity): no dedup, allows replicas of the
                #     same expr with different rs (covers the (7,4)-style
                #     wins from dedup-OFF)
                # Each half is fed by candidates whose PARENT was in that
                # half, so the two strategies evolve independently.
                half = max(1, beam_width // 2)
                h0_cands = [c for c, h in zip(candidates, candidate_halves) if h == 0]
                h1_cands = [c for c, h in zip(candidates, candidate_halves) if h == 1]
                # Bootstrap: if one half is empty (first step), feed it the
                # full candidate pool so it can populate.
                if not h0_cands:
                    h0_cands = list(candidates)
                if not h1_cands:
                    h1_cands = list(candidates)
                h0_cands.sort(key=_sort_key_weight)
                h1_cands.sort(key=_sort_key_weight)
                h0_selected = _take_top_unique_by_content(h0_cands, half)
                h1_selected = h1_cands[:half]
                # Avoid keeping the SAME candidate object in both halves —
                # would waste a beam slot. Keep half0 priority.
                h0_ids = set(id(s) for s in h0_selected)
                h1_deduped = [s for s in h1_selected if id(s) not in h0_ids]
                beam = h0_selected + h1_deduped
                weight_beam_end = len(h0_selected)
            else:
                candidates.sort(key=_sort_key)
                if dedup_beam_by_content:
                    beam = _take_top_unique_by_content(candidates, beam_width)
                else:
                    beam = candidates[:beam_width]
            # Register the new beam's exprs in the tabu set so they can't be
            # revisited on future steps. Done AFTER selection so that the
            # current step's beam can include exprs from the current
            # candidate pool that aren't already tabu.
            # Gated by env NO_TABU=1: when set, skip tabu maintenance entirely.
            import os as _os_tabu_upd
            _no_tabu_upd = bool(_os_tabu_upd.environ.get('NO_TABU'))
            if dedup_beam_by_content and not _no_tabu_upd:
                for _s in beam:
                    visited_exprs.add(frozenset(_s.expr.items()))
            t3_elapsed = time.time() - t3
            step_elapsed = time.time() - step_start

            # Optional: dump every beam state's FULL contents per step.
            # Set BEAM_DUMP_FULL=<path>. Capped at BEAM_DUMP_MAX_STEPS steps.
            # No hashes, no summaries — full expr, full subs, full resolved_subs,
            # full path, plus score and n_non_masters.
            import os as _os_bd
            _bd_path = _os_bd.environ.get('BEAM_DUMP_FULL')
            _bd_max = int(_os_bd.environ.get('BEAM_DUMP_MAX_STEPS', '5'))
            if _bd_path and step < _bd_max:
                with open(_bd_path, 'a') as _bd:
                    _bd.write(f"\n========== step={step} ==========\n")
                    for _rank, _s in enumerate(beam):
                        nm_w = sorted(
                            (weight(k)[:2] for k in get_non_masters(_s.expr, target_sector)),
                            reverse=True,
                        )
                        _bd.write(
                            f"\n--- rank={_rank}  score={_s.score:.6f}  "
                            f"nm={_s.n_non_masters}  "
                            f"mw={nm_w[0] if nm_w else (0,0)} ---\n"
                        )
                        _bd.write(f"  EXPR ({len(_s.expr)} terms):\n")
                        for k, v in sorted(_s.expr.items()):
                            _bd.write(f"    {list(k)} -> {v}\n")
                        _bd.write(f"  SUBS ({len(_s.subs)} entries):\n")
                        for k, sol in _s.subs.items():
                            _bd.write(f"    target {list(k)} -> {dict(sol)}\n")
                        _bd.write(f"  RESOLVED_SUBS ({len(_s.resolved_subs)} entries):\n")
                        for k, sol in _s.resolved_subs.items():
                            _bd.write(f"    target {list(k)} -> {dict(sol)}\n")
                        _bd.write(f"  PATH ({len(_s.path)} steps):\n")
                        for j, (t, op, d) in enumerate(_s.path):
                            _bd.write(f"    [{j}] target={list(t)} op={op} delta={list(d)}\n")
                    _bd.flush()

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
                best = min(beam, key=_sort_key_weight) if beam else None
                if best:
                    w = max_weight(best.expr, target_sector)
                    nms = get_non_masters(best.expr, target_sector)
                    if beam_sort == 'mixed':
                        n_w = min(weight_beam_end, len(beam))
                        n_t = len(beam) - n_w
                        print(f'  -> beam: {n_w} weight + {n_t} total states, best max_weight={w}, {len(nms)} non-masters', flush=True)
                    else:
                        print(f'  -> beam has {len(beam)} states, best has {best.n_non_masters} non-masters, max_weight={w}', flush=True)
                    if step < 20:
                        for nm in sorted(nms.keys(), key=lambda x: (weight(x)[0], weight(x)[1]), reverse=True):
                            print(f'    I{list(nm)} weight={weight(nm)[:2]}', flush=True)

            step += 1

            # Periodic checkpoint so an 8-CPU straggler resubmit can pick up
            # where the killed 1-CPU worker left off (rather than restarting
            # the beam search from scratch). Trigger on EITHER:
            #   - step interval (cheap-many-step runs), OR
            #   - time interval (bloated runs where each step takes minutes).
            if checkpoint_path:
                step_trigger = (checkpoint_interval and step > 0
                                and step % checkpoint_interval == 0)
                now = time.time()
                time_trigger = (checkpoint_time_seconds
                                and now - last_checkpoint_time >= checkpoint_time_seconds)
                if step_trigger or time_trigger:
                    _save_checkpoint(step)
                    last_checkpoint_time = now

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return best_solution, beam, best_weight_ever


def main():
    parser = argparse.ArgumentParser(description='Beam search reduction with classifier v5 (parallel, memory-optimized)')
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory')
    parser.add_argument('--integral', type=str, default='2,0,1,0,2,0,0',
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--checkpoint', type=str,
                        default=str(Path(__file__).parent.parent.parent / 'checkpoints/best_model.pt'),
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
    print('Beam Search Reduction with Classifier v5 (Parallel, Memory-optimized)', flush=True)
    print('  - State-level parallelization', flush=True)
    print(f'  - {args.n_workers} worker processes', flush=True)
    print('  - Running beam buffer (v13)', flush=True)
    print('=' * 70, flush=True)
    print(f'Config:', flush=True)
    for k, v in vars(args).items():
        print(f'  {k}: {v}', flush=True)
    print(flush=True)

    # Configure topology, then prime.
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
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

    model = IBPActionClassifier(
        embed_dim=ckpt_args.get('embed_dim', 256),
        n_heads=ckpt_args.get('n_heads', 4),
        n_expr_layers=ckpt_args.get('n_expr_layers', 2),
        n_cross_layers=ckpt_args.get('n_cross_layers', 2),
        n_subs_layers=ckpt_args.get('n_subs_layers', 2),
        prime=ckpt_args.get('prime', args.prime),
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions
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
