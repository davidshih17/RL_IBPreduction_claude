"""Bit-identical check: contains_sorted (nogil binsearch) vs the numpy
searchsorted membership it replaces, over many random sorted-unique int32 arrays.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from _packed_kernels import contains_sorted as C  # noqa: E402

rng = np.random.default_rng(20260617)
N = 60000
bad = 0
for _ in range(N):
    n = int(rng.integers(0, 30))
    ids = np.array(sorted(set(int(x) for x in rng.integers(0, 60, size=n))), np.int32)
    tgt = int(rng.integers(-2, 62))
    got = bool(C(ids, tgt))
    # reference: the exact numpy membership test from enumerate Phase-1b
    pos = np.searchsorted(ids, tgt)
    ref = bool(pos < len(ids) and ids[pos] == tgt)
    if got != ref:
        bad += 1
        if bad <= 3:
            print(f"  MISMATCH ids={list(ids)} tgt={tgt} got={got} ref={ref}")
print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {N} cases, {bad} mismatches")
