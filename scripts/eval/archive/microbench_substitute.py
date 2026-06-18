"""Single-core speedup of the Cython substitute_one_merge vs the numpy version,
on realistic small equations (eq ~tens of terms, rep ~9 terms, like the profile)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sailir.packed_eq as pe                       # noqa: E402
from sailir.packed_eq import PackedEq                # noqa: E402

PRIME = 1009


def rand_packed(rng, maxid, n):
    ids = np.sort(rng.choice(maxid, size=n, replace=False)).astype(np.int32)
    coeffs = rng.integers(1, PRIME, size=n).astype(np.int16)
    return PackedEq(ids, coeffs)


def main():
    rng = np.random.default_rng(1)
    cases = []
    for _ in range(2000):
        eq = rand_packed(rng, 200, int(rng.integers(15, 45)))     # cu-entry sized
        sub_id = int(eq.ids[int(rng.integers(len(eq)))])          # always present
        rep = rand_packed(rng, 200, int(rng.integers(4, 14)))     # solution sized
        keep = rep.ids != sub_id
        rep = PackedEq(np.ascontiguousarray(rep.ids[keep]),
                       np.ascontiguousarray(rep.coeffs[keep]))
        cases.append((eq, sub_id, rep))
    cy = pe._SUB_MERGE_CY
    assert cy is not None
    REPS = 30

    def run():
        for eq, sub_id, rep in cases:
            pe.substitute_one(eq, sub_id, rep.ids, rep.coeffs, PRIME)

    pe._SUB_MERGE_CY = None                          # numpy
    run()
    t = time.time()
    for _ in range(REPS):
        run()
    dt_np = time.time() - t

    pe._SUB_MERGE_CY = cy                            # cython
    run()
    t = time.time()
    for _ in range(REPS):
        run()
    dt_cy = time.time() - t

    ncalls = len(cases) * REPS
    print(f"{len(cases)} cases x {REPS} reps = {ncalls} calls")
    print(f"numpy  substitute_one: {dt_np/ncalls*1e6:6.2f} us/call  ({dt_np*1e3:.0f} ms)")
    print(f"cython substitute_one: {dt_cy/ncalls*1e6:6.2f} us/call  ({dt_cy*1e3:.0f} ms)")
    print(f"\nspeedup: {dt_np/dt_cy:.2f}x")


if __name__ == '__main__':
    main()
