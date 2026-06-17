"""Packed-cu incremental indirect-substitution (v7 Stage 2).

A faithful re-implementation of ibp_env.compute_indirect_substituted_incremental
in which the cached equations (`cu`) are PackedEq (int32 ids / int16 coeffs)
instead of {integral: coeff} dicts — this is the 17.76 GB / 83%-of-peak hog.

Control flow mirrors the dict version exactly; only the cu algebra is swapped
for the verified GF(p) kernels in packed_eq. Bit-identical by construction
(kernels proven equivalent) + verified end-to-end against the dict version.

The aux is (cu_packed, ubm, rid, iraws):
  - cu_packed : list[PackedEq]            (was list[dict])
  - ubm       : list[int]                 (unchanged — sector union bitmasks)
  - rid       : dict[(op, seed) -> idx]   (unchanged)
  - iraws     : list[(sub_int, op, shift)] (unchanged — lightweight keys)

`raw` equations stay dicts (small, from get_raw_equation) and are carried in the
result tuples for the consumer's `target in raw` check. `cached` in the result
tuples is PackedEq.

resolved_subs stay dicts at this stage (Stage 3 migrates them); we convert the
needed sub solutions to packed lazily, cached by sub_int in `packed_rs_cache`.
"""
import numpy as np

from .packed_eq import PackedEq, substitute_one, apply_resolved_subs, union_bitmask


def _get_packed_sub(sub_int, sol_dict, reg, packed_rs_cache):
    """Return (sol_ids:int32, sol_coeffs:int64) for a sub solution, caching by
    sub_int. sol_dict is the dict {integral: coeff}; safe to cache because
    resolved_subs values are immutable once produced (COW in add_sub_to_resolved).
    """
    cached = packed_rs_cache.get(sub_int)
    if cached is not None:
        return cached
    if sol_dict:
        ids = np.fromiter((reg.get_id(k) for k in sol_dict), np.int32, len(sol_dict))
        coeffs = np.fromiter((sol_dict[k] for k in sol_dict), np.int64, len(sol_dict))
    else:
        ids = np.empty(0, np.int32)
        coeffs = np.empty(0, np.int64)
    packed_rs_cache[sub_int] = (ids, coeffs)
    return ids, coeffs


def _sector_propagator_bm_from(ibp_env, target_sector):
    return ibp_env._sector_propagator_bm(target_sector)


def compute_indirect_substituted_incremental_packed(
        prev_aux, new_sub_int, new_resolved_sol, new_resolved_subs,
        ibp_t, li_t, shifts, raw_eq_cache,
        reg, prime, N_INDICES, ibp_env,
        get_raw_equation, target_sector=None):
    """Packed equivalent of compute_indirect_substituted_incremental.

    Extra params vs the dict version:
      reg               : IntegralRegistry
      prime, N_INDICES  : env scalars
      ibp_env           : the ibp_env module (for _sector_propagator_bm /
                          _entry_rejected_by_sector — sector logic unchanged)
      get_raw_equation  : ibp_env.get_raw_equation (raws stay dicts)

    Returns (result, new_aux) with cu entries as PackedEq.

    NOTE: the packed sub-solution cache is PER CALL (local), not persistent.
    resolved_subs VALUES change over the run (add_sub_to_resolved rewrites
    existing solutions via copy-on-write when a new sub is referenced), so a
    cache persisted across calls would go stale. Within one call
    new_resolved_subs is a fixed snapshot, so a local cache is both correct and
    avoids re-converting the same solution twice in this call.
    """
    packed_rs_cache = {}
    prev_cu, prev_ubm, prev_rid, prev_iraws = prev_aux

    # new sub solution in packed form (for Phase A)
    new_sub_id = reg.get_id(new_sub_int)
    sol_ids, sol_coeffs = _get_packed_sub(
        new_sub_int, new_resolved_sol, reg, packed_rs_cache)

    # ---- Phase A: substitute new_sub_int into every existing cu entry ----
    new_cu = []
    new_ubm = []
    _ts_bm = (ibp_env._sector_propagator_bm(target_sector)
              if target_sector is not None else 0)
    for old_c, old_ub in zip(prev_cu, prev_ubm):
        new_c = substitute_one(old_c, new_sub_id, sol_ids, sol_coeffs, prime)
        if new_c is old_c:
            new_cu.append(old_c)
            new_ubm.append(old_ub)
        else:
            new_ub = union_bitmask(new_c, reg)
            if (target_sector is not None
                    and ibp_env._entry_rejected_by_sector(new_ub, _ts_bm)):
                new_cu.append(PackedEq.empty())
                new_ubm.append(0)
            else:
                new_cu.append(new_c)
                new_ubm.append(new_ub)

    new_rid = dict(prev_rid)

    # ---- Phase B: add new raws coming from new_sub_int ----
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    def _get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    if new_sub_int in sub_cache:
        new_raw_list = sub_cache[new_sub_int]
    else:
        new_raw_list = []
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(new_sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = _get_raw_cached(ibp_op, seed)
                if new_sub_int in raw and raw[new_sub_int] != 0:
                    new_raw_list.append((ibp_op, shift, raw))
        sub_cache[new_sub_int] = new_raw_list

    new_iraws_for_new = [(new_sub_int, op, sh) for op, sh, _r in new_raw_list]
    new_raws_for_new = [r for _op, _sh, r in new_raw_list]

    raws_to_apply = []          # dict raws, deduped
    for (sub_int, op, sh), raw in zip(new_iraws_for_new, new_raws_for_new):
        seed = tuple(sub_int[i] - sh[i] for i in range(N_INDICES))
        rid = (op, seed)
        if rid not in new_rid:
            new_rid[rid] = len(new_cu) + len(raws_to_apply)
            raws_to_apply.append(raw)

    if raws_to_apply:
        # resolve each new raw against the full resolved_subs, in packed form
        for raw in raws_to_apply:
            praw = PackedEq.from_dict(raw, reg)
            c = _resolve_packed(praw, raw, new_resolved_subs, reg,
                                packed_rs_cache, prime)
            ub = union_bitmask(c, reg)
            if (target_sector is not None
                    and ibp_env._entry_rejected_by_sector(ub, _ts_bm)):
                new_cu.append(PackedEq.empty())
                new_ubm.append(0)
            else:
                new_cu.append(c)
                new_ubm.append(ub)

    new_iraws = prev_iraws + new_iraws_for_new

    # ---- result list: (sub_int, op, sh, raw_DICT, cu_PACKED, union_bm) ----
    raws_for_result = [None] * len(new_iraws)
    n_prev = len(prev_iraws)
    for i in range(n_prev):
        sub_int, op, sh = prev_iraws[i]
        seed = tuple(sub_int[k] - sh[k] for k in range(N_INDICES))
        raws_for_result[i] = _get_raw_cached(op, seed)
    for j, r in enumerate(new_raws_for_new):
        raws_for_result[n_prev + j] = r

    result = []
    for (sub_int, op, sh), raw in zip(new_iraws, raws_for_result):
        seed = tuple(sub_int[i] - sh[i] for i in range(N_INDICES))
        idx = new_rid[(op, seed)]
        result.append((sub_int, op, sh, raw, new_cu[idx], new_ubm[idx]))

    return result, (new_cu, new_ubm, new_rid, new_iraws)


def enumerate_valid_actions_with_indirect_cache_packed(
        target, indirect_cache, resolved_subs, ibp_t, li_t, shifts,
        filter_mode, raw_eq_cache, reg, N_INDICES):
    """Packed-cu equivalent of
    ibp_env.enumerate_valid_actions_with_indirect_cache.

    indirect_cache entries are (sub_int, ibp_op, shift, raw_DICT, cached_PACKED,
    union_bm). Phase 1a (direct) is UNCHANGED (raws are dicts). Phase 1b's
    membership `target in cached and cached[target] != 0` becomes
    `target_id in cached.ids` (binary search) — NO dict materialization, so the
    cu stays packed the whole way (this is what removes the convert-at-boundary
    transient + overhead). PackedEq never stores zero coeffs, so id-membership
    is exactly equivalent. Returns the SAME valid-action list (order included)
    as the dict version.
    """
    import numpy as _np
    from sailir.ibp_env import (
        get_raw_equation, apply_resolved_subs, cached_union_bitmask,
        sector_bitmask, action_introduces_outside_sector,
        action_introduces_higher_sector_general)

    if filter_mode == 'subsector':
        should_filter = action_introduces_outside_sector
    elif filter_mode == 'higher_only':
        should_filter = action_introduces_higher_sector_general
    elif filter_mode == 'none':
        should_filter = None
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")
    fast_subsector_filter = (filter_mode == 'subsector')
    target_bm = sector_bitmask(target) if fast_subsector_filter else None
    not_target_bm = (~target_bm) if fast_subsector_filter else None

    def _get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    valid = []
    seen = set()

    # Phase 1a: direct actions (dict raws — unchanged from the dict version)
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = _get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            if fast_subsector_filter:
                if (cached_union_bitmask(cached) & not_target_bm) != 0:
                    continue
            elif should_filter is not None and should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Phase 1b: indirect cache (PACKED cached — id-membership, no dict)
    target_id = reg.get_id(target)
    for sub_int, ibp_op, shift, raw, cached, union_bm in indirect_cache:
        if target in raw and raw[target] != 0:
            continue
        ids = cached.ids
        pos = _np.searchsorted(ids, target_id)
        if pos >= len(ids) or ids[pos] != target_id:
            continue                       # target not in cached
        if fast_subsector_filter:
            if (union_bm & not_target_bm) != 0:
                continue
        elif should_filter is not None and should_filter(cached.to_dict(reg), target):
            continue                       # rare non-subsector path only
        seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
        delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
        if (ibp_op, delta) not in seen:
            seen.add((ibp_op, delta))
            valid.append((ibp_op, delta))
    return valid


def _resolve_packed(praw, raw_dict, resolved_subs, reg, packed_rs_cache, prime):
    """apply_resolved_subs on a packed raw, converting only the subs that appear.

    Mirrors apply_resolved_subs_batch's per-raw single flat pass. raw_dict is the
    same equation as dict (cheap source of which keys are sub keys).
    """
    if not resolved_subs:
        return praw
    # build a packed_rs view for just this raw's sub-keys
    rs_packed = {}
    for k in raw_dict:
        if k in resolved_subs:
            sub_id = reg.get_id(k)
            ids, coeffs = _get_packed_sub(k, resolved_subs[k], reg, packed_rs_cache)
            rs_packed[sub_id] = (ids, coeffs)
    return apply_resolved_subs(praw, rs_packed, prime)
