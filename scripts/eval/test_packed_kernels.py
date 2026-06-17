"""Bit-identical gate for the nogil Cython substitute_one_merge vs the numpy
packed_eq.substitute_one, over many random (eq, sub_id, replacement) cases
including: sub_id present/absent, empty eq, empty rep, ids shared between eq and
rep (forcing combines), and combines that cancel to zero (dropped terms).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import sailir.packed_eq as pe                       # noqa: E402
from sailir.packed_eq import PackedEq                # noqa: E402

PRIME = 1009


def rand_packed(rng, maxid=40, nmax=12):
    n = int(rng.integers(0, nmax))
    if n == 0:
        return PackedEq.empty()
    ids = np.sort(rng.choice(maxid, size=n, replace=False)).astype(np.int32)
    coeffs = rng.integers(1, PRIME, size=n).astype(np.int16)
    return PackedEq(ids, coeffs)


def main():
    cy = pe._SUB_MERGE_CY
    if cy is None:
        print("FAIL: Cython kernel not loaded (_SUB_MERGE_CY is None)")
        sys.exit(1)
    rng = np.random.default_rng(20260617)
    n_bad = n_present = n_absent = n_zero_drop = 0
    N = 50000
    for t in range(N):
        eq = rand_packed(rng)
        rep = rand_packed(rng)
        if len(eq) and rng.random() < 0.7:
            sub_id = int(eq.ids[int(rng.integers(len(eq)))])
            n_present += 1
        else:
            sub_id = int(rng.integers(50))
            if len(eq) == 0 or sub_id not in set(int(x) for x in eq.ids):
                n_absent += 1
        # replacement must not contain sub_id (solutions never self-reference)
        keep = rep.ids != sub_id
        rep2 = PackedEq(np.ascontiguousarray(rep.ids[keep]),
                        np.ascontiguousarray(rep.coeffs[keep]))

        pe._SUB_MERGE_CY = cy
        got = pe.substitute_one(eq, sub_id, rep2.ids, rep2.coeffs, PRIME)
        pe._SUB_MERGE_CY = None
        ref = pe.substitute_one(eq, sub_id, rep2.ids, rep2.coeffs, PRIME)
        pe._SUB_MERGE_CY = cy

        if len(got) < len(eq) + len(rep2):     # something cancelled/dropped
            n_zero_drop += 1
        if not (np.array_equal(np.asarray(got.ids), np.asarray(ref.ids))
                and np.array_equal(np.asarray(got.coeffs), np.asarray(ref.coeffs))):
            n_bad += 1
            if n_bad <= 3:
                print(f"  MISMATCH t={t} sub_id={sub_id}")
                print(f"    eq.ids={list(eq.ids)} eq.co={list(eq.coeffs)}")
                print(f"    rep.ids={list(rep2.ids)} rep.co={list(rep2.coeffs)}")
                print(f"    cython ids={list(got.ids)} co={list(got.coeffs)}")
                print(f"    numpy  ids={list(ref.ids)} co={list(ref.coeffs)}")
    print(f"\n{'PASS' if n_bad == 0 else 'FAIL'}: {N} cases, {n_bad} mismatches "
          f"(present={n_present}, absent={n_absent}, with-drops={n_zero_drop})")
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
