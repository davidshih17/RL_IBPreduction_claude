"""
Measure (1) aux pickle size at production-typical state sizes, and (2)
incremental compute time vs full rebuild. This decides whether the
'attach aux to State + ship through IPC' approach is viable for the
parallel-pool architecture, or if the IPC weight kills the savings.

Run from the SAILIR_phase2 dir:
    python scripts/eval/bench_indirect_incremental.py
"""
import pickle
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
    compute_indirect_substituted,
    compute_indirect_substituted_with_aux,
    compute_indirect_substituted_incremental,
)


def random_tarsect_int(rng):
    out = []
    for i in range(ibp_env.N_INDICES):
        out.append(rng.randint(1, 3) if i < 8 else rng.randint(-3, 0))
    return tuple(out)


def random_sol(rng, n_terms):
    return {random_tarsect_int(rng): rng.randint(1, ibp_env.PRIME - 1)
            for _ in range(n_terms)}


def grow_state(env, N, seed=42):
    """Grow a synthetic state to |subs|=N by applying N random substitutions."""
    rng = random.Random(seed)
    subs = {}
    resolved_subs = {}
    for _ in range(N):
        target = random_tarsect_int(rng)
        sol = random_sol(rng, rng.randint(8, 22))
        subs[target] = sol
        resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)
    next_target = random_tarsect_int(rng)
    next_sol = random_sol(rng, 15)
    return subs, resolved_subs, next_target, next_sol


def measure(env, N_subs, n_warm=2):
    """At |subs|=N_subs measure: full compute time + aux size, then
    incremental compute time (going from N_subs to N_subs+1)."""
    subs, resolved_subs, next_target, next_sol = grow_state(env, N_subs)
    # Step N: full compute
    for _ in range(n_warm):
        compute_indirect_substituted_with_aux(
            subs, resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
    t0 = time.perf_counter()
    result_N, aux_N = compute_indirect_substituted_with_aux(
        subs, resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
    )
    t_full_N = time.perf_counter() - t0

    # Measure aux pickle size (and pickle time, since that's part of IPC cost)
    t0 = time.perf_counter()
    aux_pickle = pickle.dumps(aux_N, protocol=pickle.HIGHEST_PROTOCOL)
    t_pickle = time.perf_counter() - t0
    aux_size_bytes = len(aux_pickle)
    t0 = time.perf_counter()
    _ = pickle.loads(aux_pickle)
    t_unpickle = time.perf_counter() - t0

    # Step N+1: incremental
    # new_resolved_sol = apply_resolved_subs(next_sol, resolved_subs)
    next_resolved_sol = apply_resolved_subs(next_sol, resolved_subs)
    new_resolved_subs = add_sub_to_resolved(resolved_subs, next_target, next_sol)

    t0 = time.perf_counter()
    result_inc, aux_inc = compute_indirect_substituted_incremental(
        aux_N, next_target, next_resolved_sol, new_resolved_subs,
        env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
    )
    t_inc = time.perf_counter() - t0

    # Reference: full compute at N+1 for correctness check
    new_subs = dict(subs); new_subs[next_target] = next_sol
    result_full_Np1, _ = compute_indirect_substituted_with_aux(
        new_subs, new_resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
    )
    # Compare result lengths + a sample of (sub_int, cached) for equivalence
    n_match = sum(1 for inc, full in zip(result_inc, result_full_Np1) if inc == full)
    n_total = min(len(result_inc), len(result_full_Np1))
    correctness = n_match / n_total if n_total else 1.0

    return {
        'N': N_subs,
        't_full_ms': 1000 * t_full_N,
        't_inc_ms': 1000 * t_inc,
        'speedup': t_full_N / t_inc if t_inc > 0 else 0,
        'aux_size_kb': aux_size_bytes / 1024,
        't_pickle_ms': 1000 * t_pickle,
        't_unpickle_ms': 1000 * t_unpickle,
        'result_rows': len(result_N),
        'correctness_pct': 100 * correctness,
    }


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    print('=' * 110)
    print('compute_indirect_substituted: full rebuild vs incremental update + aux IPC cost')
    print('=' * 110)
    print(f'{"|subs|":>7} {"rows":>6} {"t_full(ms)":>11} {"t_inc(ms)":>11} {"speedup":>8} '
          f'{"aux(KB)":>9} {"pickle(ms)":>11} {"unpickle(ms)":>13} {"correct%":>9}')
    print('-' * 110)
    for N in [50, 100, 150, 200, 250, 300]:
        r = measure(env, N)
        print(f'{r["N"]:>7} {r["result_rows"]:>6} {r["t_full_ms"]:>11.1f} {r["t_inc_ms"]:>11.1f} '
              f'{r["speedup"]:>7.1f}x {r["aux_size_kb"]:>9.1f} {r["t_pickle_ms"]:>11.1f} '
              f'{r["t_unpickle_ms"]:>13.1f} {r["correctness_pct"]:>8.1f}%')
    print('=' * 110)
    print()
    print('Reading: aux(KB) × 40 beam states = per-step IPC payload if attached to State.')
    print('         (pickle+unpickle)*16 workers gives wall-time cost of shipping aux.')
    print('         Compare against (t_full - t_inc) × 40 / 16 = wall-time saved per step.')
