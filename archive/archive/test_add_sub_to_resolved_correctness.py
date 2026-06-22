"""
Correctness test for the COW-optimized add_sub_to_resolved.

Re-implements the OLD (deep-copy) version locally, then runs both on the
same random inputs and asserts identical outputs. Also re-runs the
microbenchmark to confirm the speedup.
"""
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    add_sub_to_resolved, apply_resolved_subs, set_prime,
)
# IMPORTANT: don't `from ibp_env import PRIME` — that snapshots the value at
# import time, but set_prime() rebinds the module global so the local stays
# stale. Always reference ibp_env.PRIME for consistency with the production code.


def add_sub_to_resolved_OLD(resolved_subs, target, sol):
    """Original implementation (deep-copy + post-pass cleanup) preserved
    here for the A/B correctness test."""
    resolved_sol = apply_resolved_subs(sol, resolved_subs)
    new_resolved = {k: dict(v) for k, v in resolved_subs.items()}
    new_resolved[target] = resolved_sol
    for key in resolved_subs:
        value = new_resolved[key]
        if target in value:
            coeff = value.pop(target)
            for k, v in resolved_sol.items():
                new_coeff = (coeff * v) % ibp_env.PRIME
                if k in value:
                    value[k] = (value[k] + new_coeff) % ibp_env.PRIME
                else:
                    value[k] = new_coeff
            new_resolved[key] = {k: v for k, v in value.items() if v != 0}
    return new_resolved


def gen_integral(n_indices=11, rng=None):
    rng = rng or random
    out = []
    for i in range(n_indices):
        if i < 8:
            out.append(rng.randint(1, 3))
        else:
            out.append(rng.randint(-3, 0))
    return tuple(out)


def build_rs(N, avg_value_size, rng):
    rs = {}
    keys = [gen_integral(rng=rng) for _ in range(N)]
    for k in keys:
        size = max(1, int(rng.gauss(avg_value_size, avg_value_size / 4)))
        v = {}
        for _ in range(size):
            integ = gen_integral(rng=rng)
            v[integ] = rng.randint(1, ibp_env.PRIME - 1)
        rs[k] = v
    return rs


def gen_sol(size, rng):
    sol = {}
    for _ in range(size):
        sol[gen_integral(rng=rng)] = rng.randint(1, ibp_env.PRIME - 1)
    return sol


def correctness_test(N, avg_value_size, n_trials, seed=42):
    rng_state = random.Random(seed)
    rs_init = build_rs(N, avg_value_size, rng_state)
    targets = random.Random(seed + 1).sample(list(rs_init.keys()), n_trials)
    sols = [gen_sol(random.Random(seed + 2 + i).randint(5, 20),
                    random.Random(seed + 3 + i)) for i in range(n_trials)]

    rs_a = {k: dict(v) for k, v in rs_init.items()}
    rs_b = {k: dict(v) for k, v in rs_init.items()}

    for t, sol in zip(targets, sols):
        out_old = add_sub_to_resolved_OLD(rs_a, t, sol)
        out_new = add_sub_to_resolved(rs_b, t, sol)
        if out_old != out_new:
            print(f'MISMATCH at target={t}!')
            extra_old = set(out_old) - set(out_new)
            extra_new = set(out_new) - set(out_old)
            print(f'  keys only in OLD: {len(extra_old)}; only in NEW: {len(extra_new)}')
            for k in set(out_old) & set(out_new):
                if out_old[k] != out_new[k]:
                    diff = (set(out_old[k].items()) ^ set(out_new[k].items()))
                    print(f'  diff at value [{k}]: {len(diff)} differing items')
                    if diff:
                        print(f'    sample: {list(diff)[:3]}')
                    return False
            return False
        # Roll forward both for the next iter
        rs_a = out_old
        rs_b = out_new
    return True


def speedup_bench(N, avg_value_size, n_trials, seed=42):
    rng = random.Random(seed)
    rs = build_rs(N, avg_value_size, rng)
    targets = random.Random(seed + 1).sample(list(rs.keys()), n_trials)
    sols = [gen_sol(random.Random(seed + 2 + i).randint(5, 20),
                    random.Random(seed + 3 + i)) for i in range(n_trials)]

    t0 = time.perf_counter()
    rs_a = {k: dict(v) for k, v in rs.items()}
    for t, sol in zip(targets, sols):
        rs_a = add_sub_to_resolved_OLD(rs_a, t, sol)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    rs_b = {k: dict(v) for k, v in rs.items()}
    for t, sol in zip(targets, sols):
        rs_b = add_sub_to_resolved(rs_b, t, sol)
    t_new = time.perf_counter() - t0

    return t_old, t_new


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)

    print('=' * 80)
    print('Correctness: A/B comparison OLD (deep-copy) vs NEW (COW)')
    print('=' * 80)
    for (N, M, T) in [(50, 20, 50), (100, 50, 50), (200, 50, 50), (300, 50, 30), (500, 100, 20)]:
        ok = correctness_test(N, M, T)
        print(f'  |subs|={N:<5} |value|~{M:<5} n_trials={T:<5} {"OK" if ok else "FAIL"}')

    print()
    print('=' * 80)
    print('Wall-time speedup OLD vs NEW (50 successive add_sub_to_resolved calls)')
    print('=' * 80)
    print(f'{"|subs|":<8} {"|value|":<10} {"old (ms)":>12} {"new (ms)":>12} {"speedup":>10}')
    print('-' * 60)
    for (N, M) in [(50, 20), (100, 50), (200, 50), (300, 50), (300, 100), (500, 100)]:
        t_old, t_new = speedup_bench(N, M, n_trials=50)
        sp = t_old / t_new if t_new > 0 else float('inf')
        print(f'{N:<8} {M:<10} {1000*t_old:>12.1f} {1000*t_new:>12.1f} {sp:>10.2f}x')
