"""Per-call speedup of contains_sorted (nogil binsearch) vs the numpy
searchsorted membership it replaces, on small sorted int32 arrays (the cu-entry
id sizes seen in enumerate Phase-1b)."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from _packed_kernels import contains_sorted as C  # noqa: E402

rng = np.random.default_rng(1)
# representative: tens of ids per cu entry
arrs = [np.array(sorted(set(int(x) for x in rng.integers(0, 200, size=int(rng.integers(5, 40))))),
                 np.int32) for _ in range(2000)]
tgts = [int(rng.integers(0, 200)) for _ in arrs]
REPS = 30
N = len(arrs) * REPS

def numpy_member(ids, t):
    pos = np.searchsorted(ids, t)
    return pos < len(ids) and ids[pos] == t

# warm
for a, t in zip(arrs, tgts): numpy_member(a, t); C(a, t)

t0 = time.time()
for _ in range(REPS):
    for a, t in zip(arrs, tgts): numpy_member(a, t)
dn = time.time() - t0

t0 = time.time()
for _ in range(REPS):
    for a, t in zip(arrs, tgts): C(a, t)
dc = time.time() - t0

print(f"{N} calls")
print(f"numpy searchsorted-membership: {dn/N*1e6:6.3f} us/call  ({dn*1e3:.0f} ms)")
print(f"cython contains_sorted:        {dc/N*1e6:6.3f} us/call  ({dc*1e3:.0f} ms)")
print(f"\nspeedup: {dn/dc:.2f}x")
