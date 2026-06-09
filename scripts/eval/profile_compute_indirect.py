"""
Microbenchmark for compute_indirect_substituted vs
compute_indirect_substituted_incremental.

Realistically, the late-state |subs| ~ 300; each step adds one new
substitution and otherwise reuses the prior state.

We measure:
  T_full:        cost of compute_indirect_substituted_with_aux on (subs_N+1, rs_N+1)
                 — what Phase 1 does today, for every beam state, every step.
  T_incremental: cost of compute_indirect_substituted_incremental(aux_N, ...)
                 — the proposed incremental update.

Then we extrapolate to the per-step beam cost under two schedules:
  (a) Wired into the parallel apply_actions_worker (~1500 candidates per step).
  (b) Wired post-beam-selection (~beam_width = 40 survivors per step).
"""
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    add_sub_to_resolved, apply_resolved_subs,
    compute_indirect_substituted_with_aux,
    compute_indirect_substituted_incremental,
)


def random_target_sector_integral(rng):
    out = []
    for i in range(11):
        if i < 8:
            out.append(rng.randint(1, 3))
        else:
            out.append(rng.randint(-3, 0))
    return tuple(out)


def random_sol(rng, n_terms=20):
    sol = {}
    for _ in range(n_terms):
        sol[random_target_sector_integral(rng)] = rng.randint(1, ibp_env.PRIME - 1)
    return sol


def grow_state(N, seed=42):
    """Roll resolved_subs forward by `N` random substitutions, starting empty.

    Returns (subs, resolved_subs, sample_target_for_incremental, sample_sol).
    """
    rng = random.Random(seed)
    subs = {}
    resolved_subs = {}
    for _ in range(N):
        target = random_target_sector_integral(rng)
        sol = random_sol(rng, n_terms=rng.randint(5, 25))
        # Mimic apply_action_resolved_target_only's storage step
        subs[target] = sol
        resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)
    # One extra "delta" sub for incremental test
    next_target = random_target_sector_integral(rng)
    next_sol = random_sol(rng, n_terms=rng.randint(5, 25))
    return subs, resolved_subs, next_target, next_sol


def bench_at_subs_size(env, N, n_warmup=2, n_runs=5):
    """Cost of full vs incremental at given subs size."""
    subs_before, rs_before, next_target, next_sol = grow_state(N)
    # Step N -> N+1 setup
    next_resolved_sol = apply_resolved_subs(next_sol, rs_before)
    subs_after = dict(subs_before); subs_after[next_target] = next_sol
    rs_after = add_sub_to_resolved(rs_before, next_target, next_sol)

    # Pre-step: compute aux at N (needed as input to incremental)
    result_at_N, aux_at_N = compute_indirect_substituted_with_aux(
        subs_before, rs_before, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
    )

    # --- Warmup (raw-eq cache misses) ---
    for _ in range(n_warmup):
        compute_indirect_substituted_with_aux(
            subs_after, rs_after, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )

    # --- T_full ---
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result_full, aux_full = compute_indirect_substituted_with_aux(
            subs_after, rs_after, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
    t_full = (time.perf_counter() - t0) / n_runs

    # --- T_incremental ---
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result_inc, aux_inc = compute_indirect_substituted_incremental(
            aux_at_N, next_target, next_resolved_sol, rs_after,
            env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
    t_inc = (time.perf_counter() - t0) / n_runs

    # Correctness: incremental result should match full result
    if len(result_full) != len(result_inc):
        print(f'  WARNING: result length mismatch (full={len(result_full)}, inc={len(result_inc)})')
    return t_full, t_inc, len(result_full)


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    print('=' * 110)
    print('compute_indirect_substituted: full vs incremental')
    print('=' * 110)
    print(f'{"|subs|":<8} {"result rows":>12} {"T_full (ms)":>14} {"T_inc (ms)":>14} {"speedup":>10} '
          f'{"full*40":>12} {"inc*40":>12} {"inc*1500":>12}')
    print('-' * 110)
    for N in [50, 100, 150, 200, 250, 300]:
        t_full, t_inc, n_rows = bench_at_subs_size(env, N)
        sp = t_full / t_inc if t_inc > 0 else float('inf')
        print(f'{N:<8} {n_rows:>12} {1000*t_full:>14.2f} {1000*t_inc:>14.2f} {sp:>10.2f}x '
              f'{1000*t_full*40:>11.0f}ms {1000*t_inc*40:>11.0f}ms {1000*t_inc*1500:>11.0f}ms')
    print('=' * 110)
    print('Per-step cost scenarios:')
    print('  Today (full * 40 states):                        Phase 1 cost')
    print('  Incremental, post-beam-selection (inc * 40):     Phase 1 cost if deferred to survivors')
    print('  Incremental, in worker (inc * 1500 candidates):  per-candidate aux update')
