"""Packed resolved_subs algebra (Stage 3).

resolved_subs is packed as {sub_int: PackedEq} -- the outer dict stays (one
entry per sub, ~hundreds, negligible), each solution VALUE becomes a PackedEq
(int32 ids / int16 coeffs) instead of a dict. Updates are FUNCTIONAL / copy-on-
write exactly like the dict version and like the already-shipped packed cu: a
changed solution becomes a NEW PackedEq, unchanged ones are reused (shared
across beam states). No in-place mutation, so no array-reflow problem.

These mirror beam_search_v7.add_sub_to_resolved_v5 / ibp_env.apply_resolved_subs
/ beam_search_v7.strip_passenger exactly. Bit-identical by construction (GF(p)
arithmetic via the verified packed_eq kernels; results canonical so to_dict
equals the dict path) and checked end-to-end in test_packed_rs_ops.py.

`active_id(int_id) -> bool` is supplied by the caller (= is_active(tuple, w12));
the registry/caller precomputes it since start_w12 is fixed per run.
"""
import numpy as np

from .packed_eq import PackedEq, substitute_one, apply_resolved_subs


def strip_passenger_packed(eq, active_id):
    """Keep only active-weight ids (mirrors strip_passenger). Returns eq
    unchanged (same object) when nothing is dropped, else a new PackedEq."""
    n = len(eq.ids)
    if n == 0:
        return eq
    mask = np.fromiter((active_id(int(i)) for i in eq.ids), bool, n)
    if mask.all():
        return eq
    return PackedEq(eq.ids[mask].copy(), eq.coeffs[mask].copy())


def apply_resolved_subs_packed(eq, rs_packed, reg, prime):
    """Substitute every sub-key present in `eq` with its packed solution.
    rs_packed: {sub_int: PackedEq}. One flat pass (rs solutions never contain
    sub-keys, so order is irrelevant -> bit-identical to the dict version)."""
    if not rs_packed or len(eq.ids) == 0:
        return eq
    rs_view = {}
    for i in eq.ids:
        ii = int(i)
        sol = rs_packed.get(reg.get_tuple(ii))
        if sol is not None:
            rs_view[ii] = (sol.ids, sol.coeffs)
    if not rs_view:
        return eq
    return apply_resolved_subs(eq, rs_view, prime)


def add_sub_to_resolved_packed(rs_packed, target, sol, reg, active_id, prime):
    """Packed equivalent of add_sub_to_resolved_v5.

    rs_packed: {sub_int: PackedEq}; target: sub_int (tuple); sol: PackedEq
    (raw, possibly referencing existing sub-keys / unstripped). Returns a NEW
    {sub_int: PackedEq} dict (functional; unchanged values reused/shared).
    """
    # Step 1: resolve sol against existing rs, then strip passenger
    resolved_sol = apply_resolved_subs_packed(sol, rs_packed, reg, prime)
    resolved_sol = strip_passenger_packed(resolved_sol, active_id)

    # Step 2: shallow-copy outer dict, add the new entry
    new_resolved = dict(rs_packed)
    new_resolved[target] = resolved_sol

    # Step 3: propagate target rewrite into existing entries that contain it
    target_id = reg.get_id(target)
    rep_ids = resolved_sol.ids
    rep_coeffs = resolved_sol.coeffs
    for key, old_value in rs_packed.items():
        ids = old_value.ids
        pos = int(np.searchsorted(ids, target_id))
        if pos >= len(ids) or ids[pos] != target_id:
            continue                       # target not in this solution
        # substitute_one removes the target term and splices in
        # coeff*resolved_sol, combined mod prime (== the dict Step-3 body)
        new_val = substitute_one(old_value, target_id, rep_ids, rep_coeffs, prime)
        new_val = strip_passenger_packed(new_val, active_id)
        new_resolved[key] = new_val
    return new_resolved


def to_dict_view(rs_packed, reg):
    """{sub_int: {integral: coeff}} reconstruction for verification."""
    return {k: v.to_dict(reg) for k, v in rs_packed.items()}
