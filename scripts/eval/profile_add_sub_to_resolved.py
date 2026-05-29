"""
Microbenchmark for sailir.ibp_env.add_sub_to_resolved.

Builds a synthetic resolved_subs of N entries (each value a dict of M random
target-sector integrals), then times:
  1. Full add_sub_to_resolved call (current implementation)
  2. The deep-copy step alone:  {k: dict(v) for k, v in resolved_subs.items()}
  3. The iteration step alone:  for key in resolved_subs: target in value
  4. An inverted-index "lookup hits" loop using value_to_keys

This isolates whether the inverted-index optimization the user proposed
(skip iteration over non-matching keys) addresses the dominant cost, or
whether the deep-copy is what's actually expensive at scale.
"""
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import add_sub_to_resolved, set_prime


def gen_integral(n_indices=11, rng=None):
    """Generate a plausible target-sector pentagon-box integral."""
    rng = rng or random
    out = []
    for i in range(n_indices):
        if i < 8:
            out.append(rng.randint(1, 3))
        else:
            out.append(rng.randint(-3, 0))
    return tuple(out)


def build_synthetic_resolved_subs(N, avg_value_size, rng):
    """Build a resolved_subs with N entries; each value has ~avg_value_size terms."""
    rs = {}
    keys = [gen_integral(rng=rng) for _ in range(N)]
    for k in keys:
        size = max(1, int(rng.gauss(avg_value_size, avg_value_size / 4)))
        v = {}
        for _ in range(size):
            integ = gen_integral(rng=rng)
            v[integ] = rng.randint(1, 1008)
        rs[k] = v
    return rs


def build_inverted_index(rs):
    """integral -> set of rs-keys whose value contains that integral."""
    inv = {}
    for k, v in rs.items():
        for integ in v:
            inv.setdefault(integ, set()).add(k)
    return inv


def benchmark(N, avg_value_size, n_trials, seed=42):
    rng = random.Random(seed)
    rs = build_synthetic_resolved_subs(N, avg_value_size, rng)

    # Sample `target` that exists in some values
    sample_targets = random.Random(seed + 1).sample(list(rs.keys()), n_trials)
    # And a small synthetic `sol` per call
    sols = []
    for _ in range(n_trials):
        sol = {}
        sz = random.Random(seed + 2).randint(5, 20)
        for _ in range(sz):
            sol[gen_integral(rng=rng)] = rng.randint(1, 1008)
        sols.append(sol)

    # Trial 1: full add_sub_to_resolved
    t0 = time.perf_counter()
    for target, sol in zip(sample_targets, sols):
        _ = add_sub_to_resolved(rs, target, sol)
    t_full = time.perf_counter() - t0

    # Trial 2: just deep-copy
    t0 = time.perf_counter()
    for _ in range(n_trials):
        new_resolved = {k: dict(v) for k, v in rs.items()}
        del new_resolved
    t_deepcopy = time.perf_counter() - t0

    # Trial 3: just iteration + membership check
    t0 = time.perf_counter()
    for target in sample_targets:
        hits = 0
        for k in rs:
            if target in rs[k]:
                hits += 1
    t_iter = time.perf_counter() - t0

    # Trial 4: inverted-index lookup (build once outside the loop)
    inv = build_inverted_index(rs)
    t0 = time.perf_counter()
    for target in sample_targets:
        _ = inv.get(target, set())
    t_inv = time.perf_counter() - t0

    return {
        'N': N,
        'avg_value_size': avg_value_size,
        'n_trials': n_trials,
        'full_call_ms_per_trial': 1000 * t_full / n_trials,
        'deepcopy_ms_per_trial': 1000 * t_deepcopy / n_trials,
        'iter_membership_ms_per_trial': 1000 * t_iter / n_trials,
        'inv_lookup_ms_per_trial': 1000 * t_inv / n_trials,
    }


if __name__ == '__main__':
    # Set up env (matches a typical pentagon-box run)
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)

    print('=' * 100)
    print(f'add_sub_to_resolved microbenchmark — pure Python dict arithmetic, no IBP overhead')
    print('=' * 100)
    print(f'{"|subs|":<8} {"|value|":<8} {"trials":<8} '
          f'{"full_call":>14} {"deepcopy":>14} {"iter_check":>14} {"inv_lookup":>14}')
    print(f'{"":<8} {"":<8} {"":<8} '
          f'{"(ms/call)":>14} {"(ms/call)":>14} {"(ms/call)":>14} {"(ms/call)":>14}')
    print('-' * 100)

    for N in [50, 100, 200, 300, 500]:
        for avg_value_size in [20, 50, 100]:
            r = benchmark(N, avg_value_size, n_trials=50)
            print(f'{r["N"]:<8} {r["avg_value_size"]:<8} {r["n_trials"]:<8} '
                  f'{r["full_call_ms_per_trial"]:>14.3f} '
                  f'{r["deepcopy_ms_per_trial"]:>14.3f} '
                  f'{r["iter_membership_ms_per_trial"]:>14.3f} '
                  f'{r["inv_lookup_ms_per_trial"]:>14.3f}')
    print('=' * 100)
