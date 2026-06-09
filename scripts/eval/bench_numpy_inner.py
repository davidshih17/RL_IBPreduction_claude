"""
Small test of option 2: sorted parallel arrays (keys: int64[], coeffs: int64[])
with vectorized merge, vs pure Python dict inner loop.

We assume keys have been hashed/encoded to int64 (process-stable). The cost of
that encoding is ON TOP of what's benchmarked here, but it's one-time-per-state
and dominated by the savings if this loop wins.
"""
import random
import sys
import time
import numpy as np

PRIME = 1009


# -----------------------------------------------------------------------------
# pure-Python (current production)
# -----------------------------------------------------------------------------
def py_inner(value, target, resolved_sol, PRIME):
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


# -----------------------------------------------------------------------------
# numpy version: data as (sorted keys int64, coeffs int64)
# -----------------------------------------------------------------------------
def np_inner(value_k, value_c, target_int, resolved_k, resolved_c, PRIME):
    """value_k must be sorted ascending. resolved_k sorted ascending.
    Returns new (keys, coeffs) sorted arrays with target removed and
    scaled resolved_sol merged in."""
    # locate target via binary search
    idx = np.searchsorted(value_k, target_int)
    coeff = int(value_c[idx])
    # remove target row
    rem_k = np.concatenate((value_k[:idx], value_k[idx + 1:]))
    rem_c = np.concatenate((value_c[:idx], value_c[idx + 1:]))

    # scale resolved_sol coeffs and drop zeros
    scaled = (coeff * resolved_c) % PRIME
    nz = scaled != 0
    add_k = resolved_k[nz]
    add_c = scaled[nz]

    # merge two sorted arrays — keys may overlap, summed mod PRIME, drop zeros
    # Two-pointer merge in numpy is awkward in pure-vectorized form; we use
    # the "concatenate + stable sort + reduce-on-runs" pattern.
    cat_k = np.concatenate((rem_k, add_k))
    cat_c = np.concatenate((rem_c, add_c))
    if cat_k.size == 0:
        return cat_k, cat_c
    order = np.argsort(cat_k, kind='stable')
    sk = cat_k[order]
    sc = cat_c[order]

    # group by run-length on sk
    boundaries = np.concatenate(([0], np.where(sk[1:] != sk[:-1])[0] + 1, [sk.size]))
    # for each run [boundaries[i], boundaries[i+1]) sum sc
    out_k = sk[boundaries[:-1]]
    # use add.reduceat (sums contiguous segments)
    summed = np.add.reduceat(sc, boundaries[:-1])
    out_c = summed % PRIME
    keep = out_c != 0
    return out_k[keep], out_c[keep]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def gen_int_key(rng):
    return rng.randint(0, 2**62)


def make_value_dict(rng, size):
    return {gen_int_key(rng): rng.randint(1, PRIME - 1) for _ in range(size)}


def dict_to_sorted_arrays(d):
    ks = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
    cs = np.fromiter(d.values(), dtype=np.int64, count=len(d))
    order = np.argsort(ks, kind='stable')
    return ks[order], cs[order]


def correctness():
    rng = random.Random(42)
    for trial in range(30):
        v_size = rng.randint(5, 80)
        s_size = rng.randint(5, 80)
        rng2 = random.Random(1000 + trial)
        v = make_value_dict(rng2, v_size)
        t = gen_int_key(rng2)
        v[t] = rng2.randint(1, PRIME - 1)
        s = make_value_dict(rng2, s_size)

        v_py = {k: c for k, c in v.items()}
        py_inner(v_py, t, s, PRIME)

        vk, vc = dict_to_sorted_arrays(v)
        sk, sc = dict_to_sorted_arrays(s)
        nk, nc = np_inner(vk, vc, t, sk, sc, PRIME)

        np_result = {int(nk[i]): int(nc[i]) for i in range(nk.size)}
        if v_py != np_result:
            print(f'FAIL trial {trial}: |v|={v_size} |s|={s_size}')
            print(f'  diff: py-np={set(v_py.items()) - set(np_result.items())}')
            print(f'         np-py={set(np_result.items()) - set(v_py.items())}')
            return False
    print(f'Correctness: 30/30 trials passed')
    return True


def bench(v_size, s_size, n_trials=10000):
    """Build n_trials random inputs; time the pure Python and numpy versions."""
    rng = random.Random(99)
    cases = []
    for _ in range(n_trials):
        v = make_value_dict(rng, v_size)
        t = gen_int_key(rng)
        v[t] = rng.randint(1, PRIME - 1)
        s = make_value_dict(rng, s_size)
        cases.append((v, t, s))

    # Pre-convert numpy inputs (this conversion is one-time per state in a real
    # pipeline, so we measure it separately rather than amortize it into the call)
    np_cases = []
    t0 = time.perf_counter()
    for v, t, s in cases:
        vk, vc = dict_to_sorted_arrays(v)
        sk, sc = dict_to_sorted_arrays(s)
        np_cases.append((vk, vc, t, sk, sc))
    t_convert = (time.perf_counter() - t0) / n_trials

    # py
    py_inputs = [({k: c for k, c in v.items()}, t, s) for v, t, s in cases]
    t0 = time.perf_counter()
    for v, t, s in py_inputs:
        py_inner(v, t, s, PRIME)
    t_py = (time.perf_counter() - t0) / n_trials

    # numpy
    t0 = time.perf_counter()
    for vk, vc, t, sk, sc in np_cases:
        np_inner(vk, vc, t, sk, sc, PRIME)
    t_np = (time.perf_counter() - t0) / n_trials

    return t_py, t_np, t_convert


if __name__ == '__main__':
    if not correctness():
        sys.exit(1)
    print()
    print(f'{"|value|":<8} {"|sol|":<8} {"py µs":>10} {"np µs":>10} {"speedup":>10} '
          f'{"convert µs":>12}')
    print('-' * 70)
    for v_size in [20, 30, 50, 100]:
        for s_size in [20, 30, 50, 100]:
            t_py, t_np, t_conv = bench(v_size, s_size)
            sp = t_py / t_np if t_np > 0 else 0
            print(f'{v_size:<8} {s_size:<8} {1e6*t_py:>10.2f} {1e6*t_np:>10.2f} '
                  f'{sp:>9.2f}x {1e6*t_conv:>12.2f}')
