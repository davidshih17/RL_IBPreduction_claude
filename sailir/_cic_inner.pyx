# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated inners for compute_indirect_substituted_incremental.

Two hot loops, both pure dict + int arithmetic:

  phase_a_apply(prev_cu, prev_ubm, sub_key, replacement, prime,
                bitmask_cache, n_indices) -> (new_cu, new_ubm, n_fast, n_sub)
      Walks prev_cu in lock-step with prev_ubm. For each cached:
        - if sub_key not in cached: pass through the existing object
          (preserves Python `_substitute_one_in_cached` fast-path semantics)
        - else: substitute sub_key -> replacement using
          (coeff * v) % prime arithmetic, then compute the union bitmask
          for the new cached and append it.

      Folds the per-iteration calls to _substitute_one_in_cached and
      cached_union_bitmask into native code — these two together were
      ~70% of P4_aux at production-scale state sizes.

  build_result(new_iraws, new_rid, new_cu, new_ubm) -> result list
      Replaces the per-call result-list rebuild loop in
      compute_indirect_substituted_incremental (the ~20% result_build
      line of P4_aux at late steps).
"""

def phase_a_apply(list prev_cu, list prev_ubm, tuple sub_key, dict replacement,
                   long prime, dict bitmask_cache, int n_indices):
    cdef Py_ssize_t n = len(prev_cu)
    cdef Py_ssize_t idx, i
    cdef Py_ssize_t n_fast = 0
    cdef Py_ssize_t n_sub = 0
    cdef list new_cu = [None] * n
    cdef list new_ubm = [None] * n
    cdef dict old_c, new_c
    cdef object old_ub
    cdef object k, v_obj, bm_obj
    cdef long coeff, v, nc, s, bm, union_bm
    cdef tuple k_tup

    for idx in range(n):
        old_c = <dict>prev_cu[idx]
        old_ub = prev_ubm[idx]
        if sub_key not in old_c:
            # Fast path: return same dict reference (semantic match to
            # Python _substitute_one_in_cached).
            new_cu[idx] = old_c
            new_ubm[idx] = old_ub
            n_fast += 1
            continue
        # Slow path: substitute sub_key -> replacement.
        new_c = dict(old_c)
        coeff = <long>(new_c.pop(sub_key))
        for k, v_obj in replacement.items():
            v = <long>v_obj
            nc = (coeff * v) % prime
            if nc == 0:
                continue
            if k in new_c:
                s = ((<long>new_c[k]) + nc) % prime
                if s == 0:
                    del new_c[k]
                else:
                    new_c[k] = s
            else:
                new_c[k] = nc
        new_cu[idx] = new_c
        # cached_union_bitmask for new_c, inlined.
        union_bm = 0
        for k in new_c:
            bm_obj = bitmask_cache.get(k)
            if bm_obj is None:
                k_tup = <tuple>k
                bm = 0
                for i in range(n_indices):
                    if (<long>k_tup[i]) > 0:
                        bm |= (1 << i)
                bitmask_cache[k] = bm
            else:
                bm = <long>bm_obj
            union_bm |= bm
        new_ubm[idx] = union_bm
        n_sub += 1

    return new_cu, new_ubm, n_fast, n_sub


def build_result(list new_iraws, dict new_rid, list new_cu, list new_ubm,
                  int n_indices):
    """Look up new_rid[(op, seed)] where seed = sub_int - shift componentwise.

    Switched from id(raw) keys to (op, seed) keys after the refactor that
    makes raw-equation dedup stable across env._raw_eq_cache eviction. The
    extra parameter n_indices is the index dimension (11 for pentagonbox).
    """
    cdef Py_ssize_t n = len(new_iraws)
    cdef list result = [None] * n
    cdef Py_ssize_t i, j
    cdef tuple entry
    cdef object sub_int, op, sh, raw
    cdef tuple seed
    cdef tuple si_tup, sh_tup
    cdef object key
    cdef Py_ssize_t idx
    cdef list seed_list

    for i in range(n):
        entry = <tuple>new_iraws[i]
        sub_int = entry[0]
        op = entry[1]
        sh = entry[2]
        raw = entry[3]
        si_tup = <tuple>sub_int
        sh_tup = <tuple>sh
        seed_list = [None] * n_indices
        for j in range(n_indices):
            seed_list[j] = (<long>si_tup[j]) - (<long>sh_tup[j])
        seed = tuple(seed_list)
        key = (op, seed)
        idx = <Py_ssize_t>(<long>new_rid[key])
        result[i] = (sub_int, op, sh, raw, new_cu[idx], new_ubm[idx])
    return result


def cached_union_bitmask_cy(dict cached_eq, dict bitmask_cache, int n_indices):
    """Standalone Cython cached_union_bitmask for Phase B's new entries."""
    cdef long u = 0
    cdef long bm
    cdef Py_ssize_t i
    cdef object k, bm_obj
    cdef tuple k_tup

    for k in cached_eq:
        bm_obj = bitmask_cache.get(k)
        if bm_obj is None:
            k_tup = <tuple>k
            bm = 0
            for i in range(n_indices):
                if (<long>k_tup[i]) > 0:
                    bm |= (1 << i)
            bitmask_cache[k] = bm
        else:
            bm = <long>bm_obj
        u |= bm
    return u
