"""Packed integer / GF(p) representation of SAILIR equations (v7 Stage-1 foundation).

An equation `{integral: coeff}` is a sparse vector over GF(PRIME). Here it is
stored as two parallel arrays kept SORTED-by-id (canonical):
  - ids    : int32, indices into a global IntegralRegistry (interned tuples)
  - coeffs : int16, in [1, PRIME-1]  (zeros are never stored)

~6 bytes/term vs ~250 for `dict{11-tuple: pyint}` (~40x), and the algebra is
exact, vectorizable GF(PRIME) arithmetic: with PRIME=1009 a coeff product is
<= 1008^2 ~ 1.0e6, far below int32/int64 range, so there is NO overflow and the
results are bit-identical to the dict path.

LOSSLESS: a faithful re-encoding of the same GF(p) sparse vector. Interned ids
(not bit-packed indices) impose no per-index range cap; #terms is unbounded
(variable-length arrays).

This module is pure data + kernels — no dependency on ibp_env, so it is unit-
testable in isolation. See memory: project-sailir-v7-packed-representation.
"""
import numpy as np


class IntegralRegistry:
    """Interns integral tuples <-> dense int32 ids. One instance per env.

    Also precomputes each integral's sector bitmask (OR of (1<<i) over index
    positions i with value > 0) so union_bitmask is an array gather, matching
    ibp_env.cached_union_bitmask exactly.
    """

    __slots__ = ('_to_id', '_to_tuple', '_bm')

    def __init__(self):
        self._to_id = {}        # integral tuple -> int id
        self._to_tuple = []     # int id -> integral tuple
        self._bm = []           # int id -> sector bitmask (int)

    def get_id(self, integral):
        i = self._to_id.get(integral)
        if i is None:
            i = len(self._to_tuple)
            self._to_id[integral] = i
            self._to_tuple.append(integral)
            bm = 0
            for pos in range(len(integral)):
                if integral[pos] > 0:
                    bm |= (1 << pos)
            self._bm.append(bm)
        return i

    def get_tuple(self, i):
        return self._to_tuple[i]

    def bitmask(self, i):
        return self._bm[i]

    def __len__(self):
        return len(self._to_tuple)


class PackedEq:
    """Sorted-by-id (int32 ids, int16 coeffs) sparse vector over GF(prime).

    Invariants: ids strictly ascending & unique; coeffs all nonzero in
    [1, prime-1]. The empty equation is ids/coeffs of length 0.
    """

    __slots__ = ('ids', 'coeffs')

    def __init__(self, ids, coeffs):
        self.ids = ids
        self.coeffs = coeffs

    @classmethod
    def empty(cls):
        return cls(np.empty(0, np.int32), np.empty(0, np.int16))

    @classmethod
    def from_dict(cls, d, reg):
        """Build from a {integral_tuple: coeff} dict (coeffs assumed in range)."""
        if not d:
            return cls.empty()
        keys = list(d.keys())
        ids = np.fromiter((reg.get_id(k) for k in keys), np.int64, len(keys))
        coeffs = np.fromiter((d[k] for k in keys), np.int64, len(keys))
        order = np.argsort(ids, kind='stable')
        return cls(ids[order].astype(np.int32), coeffs[order].astype(np.int16))

    def to_dict(self, reg):
        """Back to {integral_tuple: coeff} — for boundary conversion / diffing."""
        return {reg.get_tuple(int(i)): int(c)
                for i, c in zip(self.ids, self.coeffs)}

    def __len__(self):
        return len(self.ids)


def _combine(ids, coeffs, prime):
    """Group equal ids, sum coeffs mod prime, drop zeros; return sorted packed.

    `ids` need not be sorted/unique; `coeffs` is int64. Result is canonical.
    """
    if len(ids) == 0:
        return PackedEq.empty()
    order = np.argsort(ids, kind='stable')
    sids = ids[order]
    scoeffs = coeffs[order]
    # group boundaries: start index of each run of equal ids
    bounds = np.empty(len(sids), dtype=np.intp)
    bounds[0] = 0
    nb = 1
    if len(sids) > 1:
        diff = np.nonzero(np.diff(sids))[0] + 1
        bounds[1:1 + len(diff)] = diff
        nb = 1 + len(diff)
    bounds = bounds[:nb]
    summed = np.add.reduceat(scoeffs, bounds) % prime
    uniq = sids[bounds]
    nz = summed != 0
    return PackedEq(uniq[nz].astype(np.int32), summed[nz].astype(np.int16))


def substitute_one(eq, sub_id, rep_ids, rep_coeffs, prime):
    """Packed equivalent of ibp_env._substitute_one_in_cached.

    Substitute sub_id -> sum_k rep_coeffs[k] * rep_ids[k] into `eq`.
    Returns a NEW PackedEq, or `eq` UNCHANGED (same object) if sub_id is absent
    (mirrors the dict version's `return cached` fast path).

    rep_ids: int array of replacement integral ids; rep_coeffs: int array of
    replacement coeffs (already reduced mod prime).
    """
    pos = int(np.searchsorted(eq.ids, sub_id))
    if pos >= len(eq.ids) or eq.ids[pos] != sub_id:
        return eq                       # sub_id not present -> unchanged
    c = int(eq.coeffs[pos])
    # eq with the sub_id term removed
    keep = np.ones(len(eq.ids), dtype=bool)
    keep[pos] = False
    base_ids = eq.ids[keep].astype(np.int64)
    base_coeffs = eq.coeffs[keep].astype(np.int64)
    # c * replacement, reduced mod prime
    add_ids = np.asarray(rep_ids, dtype=np.int64)
    add_coeffs = (c * np.asarray(rep_coeffs, dtype=np.int64)) % prime
    all_ids = np.concatenate([base_ids, add_ids])
    all_coeffs = np.concatenate([base_coeffs, add_coeffs])
    return _combine(all_ids, all_coeffs, prime)


def apply_resolved_subs(eq, resolved_subs_packed, prime):
    """Packed equivalent of ibp_env.apply_resolved_subs (and the per-raw body of
    apply_resolved_subs_batch).

    resolved_subs_packed: {sub_id: (sol_ids ndarray, sol_coeffs ndarray)}.

    Single flat pass: expand each of `eq`'s ids that is a substitution key,
    exactly once. resolved_subs solutions never contain substitution keys, so
    one pass suffices and the application order is irrelevant (GF(p) sums
    commute) — making this bit-identical to the dict version regardless of the
    id-sorted iteration order used here.
    """
    if not resolved_subs_packed:
        return eq
    result = eq
    # snapshot the sub-key ids present in the ORIGINAL eq
    sub_ids = [int(i) for i in eq.ids if int(i) in resolved_subs_packed]
    for sid in sub_ids:
        sol_ids, sol_coeffs = resolved_subs_packed[sid]
        result = substitute_one(result, sid, sol_ids, sol_coeffs, prime)
    return result


def union_bitmask(eq, reg):
    """OR of per-integral sector bitmasks across eq's ids.

    Matches ibp_env.cached_union_bitmask exactly (registry precomputes the same
    per-integral bitmask). Returns a Python int.
    """
    u = 0
    bm = reg._bm
    for i in eq.ids:
        u |= bm[int(i)]
    return u
