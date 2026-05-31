# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated inner of enumerate_valid_actions_with_indirect_cache
Phase 1b. The Python implementation is the hot loop at late steps
(~10s/step with |indirect_cache| ~15K at depth 261).

For each (sub_int, ibp_op, shift, raw, cached, union_bm) entry in
indirect_cache, filter:
  1) skip direct actions (target in raw with non-zero coeff)
  2) skip if target not in cached or cached coeff is zero
  3) skip via sector bitmask filter
Then build seed = sub_int - shift, delta = seed - target, and append
(ibp_op, delta) to `valid` if not already seen (dedup via the `seen` set).

Mutates `seen` and `valid` in place. Returns None.

The pure-Python equivalent (currently in sailir/ibp_env.py:
enumerate_valid_actions_with_indirect_cache Phase 1b loop) does the same
operations but pays Python interpreter overhead per iteration.
"""

def phase1b_filter(list indirect_cache, tuple target, long not_target_bm,
                    set seen, list valid, int n_indices):
    cdef Py_ssize_t n = len(indirect_cache)
    cdef Py_ssize_t idx, i
    cdef tuple entry, sub_int, shift, seed, delta, key
    cdef object raw_obj, cached_obj, ibp_op
    cdef long union_bm

    for idx in range(n):
        entry = <tuple>indirect_cache[idx]
        # entry = (sub_int, ibp_op, shift, raw, cached, union_bm)
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
        # Filter 2: target must be in cached with non-zero coeff.
        if target not in <dict>cached_obj:
            continue
        if (<dict>cached_obj)[target] == 0:
            continue
        # Filter 3: sector bitmask. Reject if union_bm introduces an
        # integral outside the target sector.
        if (union_bm & not_target_bm) != 0:
            continue

        # Build seed = sub_int - shift and delta = seed - target.
        # Tuple-of-list is the fastest portable construction that still keeps
        # Python int semantics consistent with the rest of the pipeline.
        seed = tuple([sub_int[i] - shift[i] for i in range(n_indices)])
        delta = tuple([seed[i] - target[i] for i in range(n_indices)])

        key = (ibp_op, delta)
        if key not in seen:
            seen.add(key)
            valid.append(key)
