# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""nogil C kernels for the packed-equation hot loops (v7 Option-B).

substitute_one_merge: the GF(p) substitute_one as a single sorted-merge over
typed memoryviews, replacing the numpy version's searchsorted + bool-mask +
astype + concatenate + argsort + reduceat pile of small-array allocations.

Uses plain C-type memoryviews (int=int32, short=int16, long=int64 on x86-64),
so NO `cimport numpy` / numpy C headers are needed and the existing build
(_setup_cython.py) compiles it as-is. The merge loop runs nogil, so it both
speeds the single-thread path and can be driven from multiple threads.

Bit-identical to packed_eq.substitute_one by construction: same GF(p) result
(eq minus sub_id, plus c*replacement, equal-id coeffs summed mod prime, zeros
dropped), produced in canonical sorted-unique order.
"""
import numpy as np


def contains_sorted(int[::1] ids, long target_id):
    """True iff target_id is present in the ascending-sorted `ids` (nogil binary
    search). Replaces the per-entry np.searchsorted(ids, target_id) +
    bounds/equality membership test in enumerate Phase-1b, killing the numpy
    dispatch overhead. Bit-identical membership result."""
    cdef Py_ssize_t lo = 0, hi = ids.shape[0], mid
    cdef bint found = False
    if hi == 0:
        return False
    with nogil:
        while lo < hi:
            mid = (lo + hi) >> 1
            if ids[mid] < target_id:
                lo = mid + 1
            elif ids[mid] > target_id:
                hi = mid
            else:
                found = True
                break
    return found


def phase1b_packed(list indirect_cache, tuple target, long target_id,
                   long not_target_bm, set seen, list valid, int n_indices):
    """Cython Phase-1b loop for the PACKED cu (fast_subsector filter mode).

    Bit-identical to enumerate's Python Phase-1b loop: same three filters
    (1: skip direct actions where target in raw with nonzero coeff; 2: target
    must be present in cached.ids via binary search; 3: sector bitmask), same
    seed/delta construction, same dedup via `seen`, same append order into
    `valid`. Mutates `seen` and `valid` in place; returns None.

    This compiles the per-entry loop/object-handling (the ~80s self-time at
    depth) to C, while membership uses an inline binary search over the int32
    cached.ids (replacing the per-entry contains_sorted call + numpy dispatch).
    """
    cdef Py_ssize_t n = len(indirect_cache)
    cdef Py_ssize_t idx, i, lo, hi, mid, m
    cdef tuple entry, sub_int, shift, seed, delta, key
    cdef object raw_obj, cached_obj, ibp_op
    cdef long union_bm
    cdef int[::1] ids
    cdef bint found

    for idx in range(n):
        entry = <tuple>indirect_cache[idx]
        # entry = (sub_int, ibp_op, shift, raw_DICT, cached_PACKED, union_bm)
        sub_int = <tuple>entry[0]
        ibp_op = entry[1]
        shift = <tuple>entry[2]
        raw_obj = entry[3]
        cached_obj = entry[4]
        union_bm = <long>entry[5]

        # Filter 1: skip direct actions (target in raw with non-zero value).
        if target in <dict>raw_obj:
            if (<dict>raw_obj)[target] != 0:
                continue
        # Filter 2: target_id present in cached.ids (sorted int32) — binary search.
        ids = cached_obj.ids
        m = ids.shape[0]
        lo = 0
        hi = m
        found = False
        while lo < hi:
            mid = (lo + hi) >> 1
            if ids[mid] < target_id:
                lo = mid + 1
            elif ids[mid] > target_id:
                hi = mid
            else:
                found = True
                break
        if not found:
            continue
        # Filter 3: sector bitmask.
        if (union_bm & not_target_bm) != 0:
            continue

        # seed = sub_int - shift ; delta = seed - target (two-step, matches the
        # committed Python form exactly).
        seed = tuple([sub_int[i] - shift[i] for i in range(n_indices)])
        delta = tuple([seed[i] - target[i] for i in range(n_indices)])
        key = (ibp_op, delta)
        if key not in seen:
            seen.add(key)
            valid.append(key)


def substitute_one_merge(int[::1] eq_ids, short[::1] eq_coeffs, long sub_id,
                         int[::1] rep_ids, long[::1] rep_coeffs, long prime):
    """eq_ids / rep_ids ascending-sorted. Substitute sub_id -> c*replacement.

    Returns (out_ids:int32, out_coeffs:int16) canonical, or None if sub_id is
    absent (caller returns eq unchanged, matching the numpy fast path).
    """
    cdef Py_ssize_t n = eq_ids.shape[0]
    cdef Py_ssize_t m = rep_ids.shape[0]
    cdef Py_ssize_t lo = 0, hi = n, mid, sub_pos = -1
    cdef Py_ssize_t cap, i = 0, j = 0, k = 0
    cdef long c = 0, eid, rid, rc, acc
    cdef int[::1] out_ids
    cdef short[::1] out_coeffs

    # binary search sub_id in the sorted eq_ids
    while lo < hi:
        mid = (lo + hi) >> 1
        if eq_ids[mid] < sub_id:
            lo = mid + 1
        elif eq_ids[mid] > sub_id:
            hi = mid
        else:
            sub_pos = mid
            break
    if sub_pos == -1:
        return None
    c = eq_coeffs[sub_pos]

    cap = n + m
    out_ids_arr = np.empty(cap, dtype=np.int32)
    out_coeffs_arr = np.empty(cap, dtype=np.int16)
    out_ids = out_ids_arr
    out_coeffs = out_coeffs_arr

    # two-pointer merge of (eq without sub_pos) and (c * rep mod prime),
    # combining equal ids mod prime, dropping zeros. Both inputs sorted ->
    # output sorted/unique/nonzero = canonical. All coeffs positive in
    # [1,prime-1] and c>0, so C % is non-negative (matches Python %).
    with nogil:
        while True:
            while i < n and i == sub_pos:
                i += 1
            if i >= n and j >= m:
                break
            if i >= n:                                   # rep tail only
                rc = (c * rep_coeffs[j]) % prime
                if rc != 0:
                    out_ids[k] = rep_ids[j]
                    out_coeffs[k] = <short> rc
                    k += 1
                j += 1
                continue
            if j >= m:                                   # eq tail only
                out_ids[k] = eq_ids[i]
                out_coeffs[k] = eq_coeffs[i]
                k += 1
                i += 1
                continue
            eid = eq_ids[i]
            rid = rep_ids[j]
            if eid < rid:
                out_ids[k] = eq_ids[i]
                out_coeffs[k] = eq_coeffs[i]
                k += 1
                i += 1
            elif rid < eid:
                rc = (c * rep_coeffs[j]) % prime
                if rc != 0:
                    out_ids[k] = rep_ids[j]
                    out_coeffs[k] = <short> rc
                    k += 1
                j += 1
            else:                                        # equal id -> combine
                acc = (eq_coeffs[i] + c * rep_coeffs[j]) % prime
                if acc != 0:
                    out_ids[k] = eq_ids[i]
                    out_coeffs[k] = <short> acc
                    k += 1
                i += 1
                j += 1

    # Return COMPACT copies, not slices: out_*_arr were over-allocated to the
    # max size cap=n+m, and a slice view would pin that whole buffer in every
    # stored cu entry (k can be << n+m after GF cancellation). .copy() gives an
    # exactly-k array and lets the over-allocated buffer be freed.
    return out_ids_arr[:k].copy(), out_coeffs_arr[:k].copy()


def phaseA_substitute_all(list prev_cu, list prev_ubm, long sub_id,
                          int[::1] rep_ids, long[::1] rep_coeffs, long prime,
                          long[::1] reg_bm, object PackedEq_cls):
    """Fused Phase A of attach_aux: substitute sub_id -> rep into EVERY cu entry
    in one compiled loop. Bit-identical to the Python loop

        for old_c, old_ub in zip(prev_cu, prev_ubm):
            new_c = substitute_one(old_c, sub_id, rep_ids, rep_coeffs, prime)
            if new_c is old_c: keep old_c, old_ub
            else: keep new_c, union_bitmask(new_c)

    but the per-entry substitute_one wrapper, the intermediate PackedEq for the
    merge, and the per-entry union_bitmask Python loop are all inlined into C.
    Assumes target_sector is None (the incremental caller's only mode); the
    sector-rejection case is left to the Python fallback.

    Memory: ONE reusable scratch buffer (largest entry + rep), reused across all
    entries; only CHANGED entries allocate a compact .copy() (which they must, it
    is stored). No per-entry buffer growth.

    Returns (new_cu list, new_ubm list).
    """
    cdef Py_ssize_t n = len(prev_cu), m = rep_ids.shape[0]
    cdef Py_ssize_t k, en, lo, hi, mid, sub_pos, i, j, kk, cap, max_en = 0
    cdef long c, eid, rid, rc, acc, ubm
    cdef int[::1] eq_ids
    cdef short[::1] eq_coeffs
    cdef int[::1] out_ids
    cdef short[::1] out_coeffs
    cdef object old_c

    new_cu = [None] * n
    new_ubm = [0] * n

    for k in range(n):
        en = (<object>prev_cu[k]).ids.shape[0]
        if en > max_en:
            max_en = en
    cap = max_en + m
    if cap < 1:
        cap = 1
    out_ids_arr = np.empty(cap, dtype=np.int32)
    out_coeffs_arr = np.empty(cap, dtype=np.int16)
    out_ids = out_ids_arr
    out_coeffs = out_coeffs_arr

    for k in range(n):
        old_c = prev_cu[k]
        eq_ids = old_c.ids
        eq_coeffs = old_c.coeffs
        en = eq_ids.shape[0]
        lo = 0
        hi = en
        sub_pos = -1
        while lo < hi:
            mid = (lo + hi) >> 1
            if eq_ids[mid] < sub_id:
                lo = mid + 1
            elif eq_ids[mid] > sub_id:
                hi = mid
            else:
                sub_pos = mid
                break
        if sub_pos == -1:                       # sub_id absent -> entry unchanged
            new_cu[k] = old_c
            new_ubm[k] = prev_ubm[k]
            continue
        c = eq_coeffs[sub_pos]
        i = 0
        j = 0
        kk = 0
        ubm = 0
        while True:
            while i < en and i == sub_pos:
                i += 1
            if i >= en and j >= m:
                break
            if i >= en:
                rc = (c * rep_coeffs[j]) % prime
                if rc != 0:
                    out_ids[kk] = rep_ids[j]
                    out_coeffs[kk] = <short> rc
                    ubm |= reg_bm[rep_ids[j]]
                    kk += 1
                j += 1
                continue
            if j >= m:
                out_ids[kk] = eq_ids[i]
                out_coeffs[kk] = eq_coeffs[i]
                ubm |= reg_bm[eq_ids[i]]
                kk += 1
                i += 1
                continue
            eid = eq_ids[i]
            rid = rep_ids[j]
            if eid < rid:
                out_ids[kk] = eq_ids[i]
                out_coeffs[kk] = eq_coeffs[i]
                ubm |= reg_bm[eid]
                kk += 1
                i += 1
            elif rid < eid:
                rc = (c * rep_coeffs[j]) % prime
                if rc != 0:
                    out_ids[kk] = rep_ids[j]
                    out_coeffs[kk] = <short> rc
                    ubm |= reg_bm[rid]
                    kk += 1
                j += 1
            else:
                acc = (eq_coeffs[i] + c * rep_coeffs[j]) % prime
                if acc != 0:
                    out_ids[kk] = eq_ids[i]
                    out_coeffs[kk] = <short> acc
                    ubm |= reg_bm[eid]
                    kk += 1
                i += 1
                j += 1
        new_cu[k] = PackedEq_cls(out_ids_arr[:kk].copy(),
                                 out_coeffs_arr[:kk].copy())
        new_ubm[k] = ubm
    return new_cu, new_ubm
