"""
Benchmark Cython inner loop vs pure Python equivalent.
Realistic conditions: |value| in {20, 50, 100}, |resolved_sol| in {20, 50, 100}.
"""
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import _add_sub_inner

PRIME = 1009


def py_inner(value, target, resolved_sol, PRIME):
    """Pure Python equivalent (matches add_sub_to_resolved current inner loop)."""
    coeff = value.pop(target)
    for k, v in resolved_sol.items():
        new_coeff = (coeff * v) % PRIME
        if new_coeff == 0:
            continue
        if k in value:
            s = (value[k] + new_coeff) % PRIME
            if s == 0:
                del value[k]
            else:
                value[k] = s
        else:
            value[k] = new_coeff
    return value


def cy_inner(value, target, resolved_sol, PRIME):
    return _add_sub_inner.apply_sub_inner(value, target, resolved_sol, PRIME)


def gen_tuple(rng):
    return tuple(rng.randint(-3, 3) for _ in range(11))


def make_value(rng, size):
    return {gen_tuple(rng): rng.randint(1, PRIME - 1) for _ in range(size)}


def make_resolved_sol(rng, size):
    return {gen_tuple(rng): rng.randint(1, PRIME - 1) for _ in range(size)}


def correctness():
    rng = random.Random(42)
    for trial in range(50):
        v_size = rng.randint(5, 80)
        s_size = rng.randint(5, 80)
        rng2 = random.Random(1000 + trial)
        value_py = make_value(rng2, v_size)
        # add target inside
        target = gen_tuple(rng2)
        value_py[target] = rng2.randint(1, PRIME - 1)
        value_cy = {k: v for k, v in value_py.items()}
        rsol = make_resolved_sol(rng2, s_size)
        py_inner(value_py, target, rsol, PRIME)
        cy_inner(value_cy, target, rsol, PRIME)
        if value_py != value_cy:
            print(f'FAIL trial {trial}: |v|={v_size} |s|={s_size}')
            print(f'  diff: {set(value_py.items())^set(value_cy.items())}')
            return False
    print(f'Correctness: 50/50 trials passed')
    return True


def bench_size(v_size, s_size, n_trials=10000):
    rng = random.Random(99)
    # Build fresh trial inputs each call so cache doesn't trivialize
    values = []
    targets = []
    sols = []
    for _ in range(n_trials):
        v = make_value(rng, v_size)
        t = gen_tuple(rng)
        v[t] = rng.randint(1, PRIME - 1)
        values.append(v)
        targets.append(t)
        sols.append(make_resolved_sol(rng, s_size))

    # Pure Python
    t0 = time.perf_counter()
    for v, t, s in zip(values, targets, sols):
        py_inner(v, t, s, PRIME)
    t_py = time.perf_counter() - t0

    # Cython
    rng = random.Random(99)
    values2 = []
    targets2 = []
    sols2 = []
    for _ in range(n_trials):
        v = make_value(rng, v_size)
        t = gen_tuple(rng)
        v[t] = rng.randint(1, PRIME - 1)
        values2.append(v)
        targets2.append(t)
        sols2.append(make_resolved_sol(rng, s_size))

    t0 = time.perf_counter()
    for v, t, s in zip(values2, targets2, sols2):
        cy_inner(v, t, s, PRIME)
    t_cy = time.perf_counter() - t0

    return t_py / n_trials, t_cy / n_trials


if __name__ == '__main__':
    if not correctness():
        sys.exit(1)
    print()
    print(f'{"|value|":<8} {"|sol|":<8} {"py µs/call":>12} {"cy µs/call":>12} {"speedup":>10}')
    print('-' * 60)
    for v_size in [20, 50, 100]:
        for s_size in [20, 50, 100]:
            t_py, t_cy = bench_size(v_size, s_size)
            sp = t_py / t_cy if t_cy > 0 else 0
            print(f'{v_size:<8} {s_size:<8} {1e6*t_py:>12.2f} {1e6*t_cy:>12.2f} {sp:>9.2f}x')
