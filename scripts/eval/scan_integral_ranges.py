"""Scan a beam_search_v7 checkpoint for the per-position power ranges of all
live integrals (the 11-tuple dict keys in cu / resolved_subs / expr / sub_accum
/ tabu). Decides the bit-packing-into-int32 feasibility question:

  Does  PROD_i (max_i - min_i + 1)  fit under 2^32 ?

If yes, a stateless variable-radix int32 codec is possible (drop the registry).
If no, int32 needs the registry (count-based ids) or int64 bit-packing.

The checkpoint is registry-INDEPENDENT (aux converted to dicts), so we only need
a matching State_v5 namedtuple to unpickle; no torch / model / topology import.

Usage: python scan_integral_ranges.py <ckpt.pkl>
Output is the full report; nothing filtered.
"""
import pickle
import sys
from collections import namedtuple

N = 11  # integral tuple length (pentagonbox)

# Matching namedtuple so the pickle's State_v5 instances reconstruct without
# importing beam_search_v7 (which would pull in torch).
State_v5 = namedtuple(
    'State_v5',
    ['expr', 'resolved_subs', 'sub_accum', 'score', 'path',
     'n_non_masters', 'max_w12', 'total_w12', 'aux_flat'])


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'State_v5':
            return State_v5
        return super().find_class(module, name)


def is_integral(k):
    return (isinstance(k, tuple) and len(k) == N
            and all(isinstance(x, int) and not isinstance(x, bool) for x in k))


def main():
    path = sys.argv[1]
    print(f"loading {path} ...", flush=True)
    with open(path, 'rb') as f:
        obj = _Unpickler(f).load()
    print("loaded; walking ...", flush=True)

    mins = [10**9] * N
    maxs = [-10**9] * N
    distinct = set()
    n_keys_seen = 0
    n_other_11tuples = 0  # 11-int-tuples NOT used as dict key / set elem (e.g. deltas)

    visited = set()
    stack = [obj]
    while stack:
        o = stack.pop()
        oid = id(o)
        if isinstance(o, (dict, list, tuple, set, frozenset)):
            if oid in visited:
                continue
            visited.add(oid)
        if isinstance(o, dict):
            for k, v in o.items():
                if is_integral(k):
                    distinct.add(k)
                    n_keys_seen += 1
                    for i in range(N):
                        ki = k[i]
                        if ki < mins[i]:
                            mins[i] = ki
                        if ki > maxs[i]:
                            maxs[i] = ki
                else:
                    stack.append(k)
                stack.append(v)
        elif isinstance(o, (set, frozenset)):
            for x in o:
                if is_integral(x):
                    distinct.add(x)
                    n_keys_seen += 1
                    for i in range(N):
                        xi = x[i]
                        if xi < mins[i]:
                            mins[i] = xi
                        if xi > maxs[i]:
                            maxs[i] = xi
                else:
                    stack.append(x)
        elif isinstance(o, (list, tuple)):
            if is_integral(o):
                n_other_11tuples += 1
            for x in o:
                stack.append(x)

    print("\n==== per-position power ranges (integrals = dict keys / set elems) ====")
    prod_bits = 0.0
    import math
    for i in range(N):
        rng = maxs[i] - mins[i] + 1 if maxs[i] >= mins[i] else 0
        bits = math.log2(rng) if rng > 0 else 0.0
        prod_bits += bits
        print(f"  pos {i:2d}: min={mins[i]:>4d} max={maxs[i]:>4d}  "
              f"range={rng:>3d}  bits={bits:.2f}")

    print(f"\n  distinct live integrals : {len(distinct):,}")
    print(f"  integral-key occurrences: {n_keys_seen:,}")
    print(f"  other 11-int-tuples (deltas/path, not counted): {n_other_11tuples:,}")
    print(f"\n  SUM of per-position bits (variable-radix id width) = {prod_bits:.2f} bits")
    print(f"  product of ranges = 2^{prod_bits:.2f}")
    print(f"  int32 budget = 32 bits ; int64 = 64 bits")
    fits32 = prod_bits <= 32.0
    fits64 = prod_bits <= 64.0
    print(f"\n  ==> variable-radix bit-pack FITS int32: {fits32}")
    print(f"  ==> variable-radix bit-pack FITS int64: {fits64}")
    print(f"  (count of distinct live integrals needs "
          f"{math.log2(len(distinct)) if distinct else 0:.1f} bits "
          f"-> registry int32 ok: {len(distinct) < 2**31})")


if __name__ == '__main__':
    main()
