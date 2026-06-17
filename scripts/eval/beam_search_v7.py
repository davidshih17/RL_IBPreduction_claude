#!/usr/bin/env python
"""v6 beam search: macro-beam architecture on top of v5 strip-passenger.

Built from v5 (`beam_search_v5.py`). v5 wastes beam capacity because:
  - The model only sees expr + RS_KEYS (not RS values; see prepare_batched
    _input_v5_dummy). When 40 beam states share the same expr_fp and share
    ~249/250 RS keys (as empirical inspection of probe_84_v5_lazyrs at step
    250 confirmed), the model returns essentially the same scores for all
    40 — they all pick the same top-K action.
  - Tabu adds a tiny bit of artificial variation but the beam still collapses
    to 1-3 distinct exprs / 40 slots.

v6 changes:
  1. At the top of each step, group beam by `expr_fp = frozenset(expr.items())`
     and keep ONE representative per macro (the highest-score variant).
  2. Run the model ONCE per (macro, target) pair — not per (state, target).
  3. Expand top-K actions per macro into candidates; per-expr cap during
     beam selection enforces real expr diversity in the next beam.

What we keep from v5:
  - strip-passenger semantics (apply_substitution_v5, add_sub_to_resolved_v5)
  - aux_flat / LAZY_RS / iraws-keep-first / tabu
  - --beam-sort {weight,mixed}
  - any() termination on first state with nm=0
"""
import argparse
import math
import os
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
import torch

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, apply_substitution_target_only,
    enumerate_valid_actions_with_indirect_cache, compute_indirect_substituted,
    compute_indirect_substituted_with_aux,
    weight, is_master, solve_ibp_for,
)


def _aux_to_result(aux_flat, env):
    """Reconstruct compute_indirect_substituted result list from aux tuple.

    iraws entries are 3-tuples (sub_int, op, shift). Raws are looked up
    fresh from env._raw_eq_cache (recomputed on miss) — only the cache
    persistently holds raw equations, so eviction can free memory.
    """
    cached_unique, union_bms, raw_id_to_idx, indirect_raws = aux_flat
    if not indirect_raws:
        return []
    n_indices = len(indirect_raws[0][0])
    result = []
    for sub_int, ibp_op, shift in indirect_raws:
        seed = tuple(sub_int[i] - shift[i] for i in range(n_indices))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        idx = raw_id_to_idx[(ibp_op, seed)]
        # v7 Stage 2b: cu entries stay PackedEq in the result — the packed
        # enumerate consumes them directly (id-membership searchsorted), so NO
        # dict conversion here (that per-step transient is what capped the
        # memory win and cost the time).
        result.append((sub_int, ibp_op, shift, raw,
                       cached_unique[idx], union_bms[idx]))
    return result


_PICKLABLE_AUX_MARKER = '__v5_aux_v2__'


def _aux_to_picklable(aux_flat):
    """rid is (op, seed)-keyed; iraws are 3-tuples (sub_int, op, shift).
    All pickle-stable. Append marker for format detection on load.
    """
    if aux_flat is None:
        return None
    cu, ubm, rid, iraws = aux_flat
    # v7: PackedEq cu -> dict cu so the checkpoint is registry-independent
    # (re-interned on load via _aux_from_picklable).
    if cu and isinstance(cu[0], PackedEq):
        cu = [c.to_dict(_V7_REGISTRY) for c in cu]
    return (cu, ubm, rid, iraws, _PICKLABLE_AUX_MARKER)


def _aux_from_picklable(aux_flat, env=None):
    """rid stays (op, seed)-keyed; iraws are 3-tuples (no raw stored)."""
    if aux_flat is None:
        return None
    if len(aux_flat) == 5 and aux_flat[4] == _PICKLABLE_AUX_MARKER:
        cu, ubm, rid, iraws, _ = aux_flat
        # v7: re-intern dict cu -> PackedEq
        if cu and isinstance(cu[0], dict):
            cu = [PackedEq.from_dict(c, _V7_REGISTRY) for c in cu]
        return (cu, ubm, rid, iraws)
    # Legacy 4-tuple format — assume same-process; just pass through.
    return aux_flat


def _prune_aux_by_recency(aux_flat, recent_sub_ints):
    """Drop iraws entries whose sub_int (anchor) isn't in recent_sub_ints.
    Rebuild cu/ubm/rid from the kept raws.

    Args:
        aux_flat: (cached_unique, union_bms, raw_key_to_idx, indirect_raws)
            where raw_key_to_idx is keyed by (op, seed) tuples and
            indirect_raws entries are 3-tuples (sub_int, op, shift).
        recent_sub_ints: iterable of sub_int tuples to KEEP.

    Returns the pruned aux tuple (unchanged tuple if nothing dropped).
    """
    cu, ubm, rid, iraws = aux_flat
    if not iraws:
        return aux_flat
    keep_set = set(recent_sub_ints)
    new_iraws = [e for e in iraws if e[0] in keep_set]
    if len(new_iraws) == len(iraws):
        return aux_flat  # nothing to drop
    # Rebuild cu/ubm/rid: keep only entries referenced by surviving iraws.
    # Key is (op, seed) where seed = sub_int - shift componentwise.
    referenced = set()
    for sub_int, op, shift in new_iraws:
        n = len(sub_int)
        seed = tuple(sub_int[i] - shift[i] for i in range(n))
        referenced.add((op, seed))
    new_cu = []
    new_ubm = []
    new_rid = {}
    for key, old_idx in rid.items():
        if key in referenced:
            new_rid[key] = len(new_cu)
            new_cu.append(cu[old_idx])
            new_ubm.append(ubm[old_idx])
    return (new_cu, new_ubm, new_rid, new_iraws)
# DO NOT `from ... import PRIME` — set_prime() rebinds the module attr but
# not local bindings. Always read ibp_env.PRIME so we pick up the configured value.
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_non_masters, get_sector_mask, filter_to_sector

# ── v7: packed-int/GF(p) cu representation ──────────────────────────────────
# cu entries are stored as PackedEq (int32 ids / int16 coeffs) in the aux —
# the persistent 40x-smaller form. Consumers (enumerate / solve) still see
# dicts: _aux_to_result converts the (pruned, one-state-at-a-time) cu to dicts
# transiently. One IntegralRegistry per worker process (module global).
from sailir.packed_eq import IntegralRegistry, PackedEq
from sailir import packed_cu as _packed_cu

_V7_REGISTRY = None          # IntegralRegistry, set in main()
_V7_PACKED_RS_CACHE = None   # {sub_int: (ids, coeffs)}, grown lazily


def _v7_pack_cu(dict_cu):
    """Convert a list of {integral: coeff} dict cu entries -> list[PackedEq]."""
    return [PackedEq.from_dict(c, _V7_REGISTRY) for c in dict_cu]


def _v7_aux_pack(aux_flat):
    """Pack a dict-cu aux (from a dict compute_indirect) into packed-cu aux."""
    cu, ubm, rid, iraws = aux_flat
    return (_v7_pack_cu(cu), ubm, rid, iraws)


def _v7_as_dict(eq):
    """A single cu entry -> dict (convert if PackedEq) for solve_ibp_for."""
    return eq.to_dict(_V7_REGISTRY) if isinstance(eq, PackedEq) else eq


# ============================================================================
# v5 State
# ============================================================================

State_v5 = namedtuple(
    'State_v5',
    ['expr', 'resolved_subs', 'sub_accum',
     'score', 'path', 'n_non_masters',
     'max_w12', 'total_w12',  # cached sort keys, computed once at construction
     'aux_flat'],  # (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
                   # or None to mean "rebuild from scratch on use"
)
State_v5.__new__.__defaults__ = (None, None, None)  # max_w12, total_w12, aux_flat defaults


# ============================================================================
# Weight-active partition
# ============================================================================

def is_active(integral, start_w12):
    """True iff integral's weight (w1, w2) is lex-≥ start_w12."""
    w = weight(integral)
    return (w[0], w[1]) >= start_w12


def strip_passenger(d, start_w12):
    """Return new dict containing only active-weight entries."""
    return {k: v for k, v in d.items() if is_active(k, start_w12)}


# ============================================================================
# v5 substitution math (Option F + weight strip)
# ============================================================================

def apply_substitution_v5(expr_t, sub_accum, sub_int, sol,
                          target_sector, start_w12):
    """Apply substitution; route results by (sector, weight):
      - active target-sector terms → new_expr_t
      - passenger target-sector terms (below start_w) → discarded (recovered
        by replay; their cumulative effect on the final expression is exactly
        what the path encodes)
      - sub-sector terms → new_sub_accum (Option F)
    """
    if sub_int not in expr_t:
        return expr_t, sub_accum
    coeff = expr_t[sub_int]
    new_expr_t = {k: v for k, v in expr_t.items() if k != sub_int}
    new_sub_accum = dict(sub_accum) if sub_accum else {}
    nd = ibp_env.N_DENOMINATORS
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % ibp_env.PRIME
        if new_coeff == 0:
            continue
        # Sector check
        in_target_sector = True
        for i in range(nd):
            if (1 if integral[i] > 0 else 0) != target_sector[i]:
                in_target_sector = False
                break
        if not in_target_sector:
            # passenger sub-sector — accumulate
            if integral in new_sub_accum:
                s = (new_sub_accum[integral] + new_coeff) % ibp_env.PRIME
                if s == 0:
                    del new_sub_accum[integral]
                else:
                    new_sub_accum[integral] = s
            else:
                new_sub_accum[integral] = new_coeff
            continue
        # Target-sector — check weight
        if not is_active(integral, start_w12):
            # passenger lower-weight in target sector — DISCARDED from active beam
            continue
        if integral in new_expr_t:
            s = (new_expr_t[integral] + new_coeff) % ibp_env.PRIME
            if s == 0:
                del new_expr_t[integral]
            else:
                new_expr_t[integral] = s
        else:
            new_expr_t[integral] = new_coeff
    return new_expr_t, new_sub_accum


def add_sub_to_resolved_v5(resolved_subs, target, sol, start_w12):
    """Like add_sub_to_resolved but keeps every value dict stripped of
    passenger-weight entries. `sol` may be unstripped; we strip after every
    write.
    """
    # Step 1: resolve sol against existing RS, then strip passenger
    resolved_sol = apply_resolved_subs(sol, resolved_subs)
    resolved_sol = strip_passenger(resolved_sol, start_w12)

    # Step 2: shallow-copy outer dict, add new entry
    new_resolved = dict(resolved_subs)
    new_resolved[target] = resolved_sol

    # Step 3: propagate target rewrite into existing entries that contain it
    for key, old_value in resolved_subs.items():
        if target not in old_value:
            continue
        value = dict(old_value)
        coeff = value.pop(target)
        for k, v in resolved_sol.items():
            new_coeff = (coeff * v) % ibp_env.PRIME
            if k in value:
                s = (value[k] + new_coeff) % ibp_env.PRIME
                if s == 0:
                    del value[k]
                else:
                    value[k] = s
            elif new_coeff != 0:
                value[k] = new_coeff
        # Strip again (resolved_sol is already stripped but safety)
        new_resolved[key] = strip_passenger(value, start_w12)
    return new_resolved


def apply_action_v5(state, target, ibp_op, delta, action_prob,
                    env, target_sector, start_w12,
                    use_incremental_aux=True,
                    lazy_rs=True):
    """Apply one (target, ibp_op, delta) action to a v5 State.
    Returns a (new_state, sol) tuple, or None on failure.

    sol is the raw IBP sol (active+passenger) that the action produces.
    Caller stores sol alongside (parent, target) so survivors can have
    their resolved_subs materialized post-selection.

    With lazy_rs=True (default), child.resolved_subs is left as None.
    add_sub_to_resolved_v5 is the expensive per-RS-value propagation step.
    Materialize for survivors only via `_materialize_lazy_rs()` after
    beam selection.
    """
    n_idx = ibp_env.N_INDICES
    seed = tuple(target[i] + delta[i] for i in range(n_idx))
    # v7 Stage 2b: REUSE the already-resolved cu entry when present (indirect
    # actions). compute_indirect's invariant (target_sector=None, entry not
    # pruned): cu[(ibp_op,seed)] == apply_resolved_subs(raw, resolved_subs), so
    # this is bit-identical and skips the re-resolution that grows with
    # |resolved_subs|. Falls back to recompute on a cu miss (direct/pruned).
    cached = None
    if state.aux_flat is not None:
        _idx = state.aux_flat[2].get((ibp_op, seed))
        if _idx is not None:
            cached = _v7_as_dict(state.aux_flat[0][_idx])
    if cached is None:
        raw = env.get_raw_equation_cached(ibp_op, seed)
        cached = apply_resolved_subs(raw, state.resolved_subs)
    if target not in cached or cached[target] == 0:
        return None
    sol = solve_ibp_for(cached, target)
    if sol is None:
        return None
    new_expr, new_sub_accum = apply_substitution_v5(
        state.expr, state.sub_accum, target, sol, target_sector, start_w12,
    )

    if lazy_rs:
        new_rs = None  # to be materialized post-selection for survivors only
    else:
        new_rs = add_sub_to_resolved_v5(
            state.resolved_subs, target, sol, start_w12,
        )

    new_path = state.path + [(target, ibp_op, delta)]
    new_score = state.score + math.log(action_prob + 1e-10)
    nm = get_non_masters(new_expr, target_sector)
    if nm:
        w_list = [(weight(k)[0], weight(k)[1]) for k in nm]
        mw = max(w_list)
        tw = (sum(w[0] for w in w_list), sum(w[1] for w in w_list))
    else:
        mw = (0, 0)
        tw = (0, 0)
    child = State_v5(
        expr=new_expr,
        resolved_subs=new_rs,
        sub_accum=new_sub_accum,
        score=new_score,
        path=new_path,
        n_non_masters=len(nm),
        max_w12=mw,
        total_w12=tw,
        aux_flat=None,
    )
    return child, sol


def _materialize_lazy_rs(child, parent, target, sol, start_w12):
    """Materialize child.resolved_subs via add_sub_to_resolved_v5(parent.RS, target, sol).
    Lossless — same inputs/outputs as if apply_action_v5 had run with lazy_rs=False.
    """
    if child.resolved_subs is not None:
        return child  # already materialized
    new_rs = add_sub_to_resolved_v5(
        parent.resolved_subs, target, sol, start_w12,
    )
    return child._replace(resolved_subs=new_rs)


def _attach_incremental_aux(child, parent, target, env, target_sector,
                            use_exprkeyed=True, iraws_window=None,
                            iraws_keep_first=None):
    """Compute child's aux_flat from parent's aux_flat + the action target.
    Returns a new State_v5 with aux_flat set. Skips if parent's aux is None.

    use_exprkeyed: if True (default), use the bounded-anchor exprkeyed delta
        (iraws bounded by |useful K| ≈ |expr_nm|, not by |RS|). Otherwise use
        the depth-keyed compute_indirect_substituted_incremental.
    """
    if parent.aux_flat is None:
        return child  # can't do incremental; will rebuild from scratch on use
    new_resolved_sol = child.resolved_subs[target]
    # v7: always the PACKED depth-keyed path (exprkeyed is a dead end — it
    # explodes per-step time at depth). parent.aux_flat carries PackedEq cu.
    _result, new_aux = _packed_cu.compute_indirect_substituted_incremental_packed(
        parent.aux_flat, target, new_resolved_sol, child.resolved_subs,
        env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        _V7_REGISTRY, ibp_env.PRIME, ibp_env.N_INDICES,
        ibp_env, ibp_env.get_raw_equation, target_sector=None)
    # Optional: prune iraws to first N sub_ints (anchor-as-bootstrap) AND/OR
    # last N sub_ints (recency).
    if iraws_keep_first is not None or iraws_window is not None:
        keys = list(child.resolved_subs.keys())
        keep = set()
        if iraws_keep_first is not None:
            keep.update(keys[:iraws_keep_first])
        if iraws_window is not None:
            keep.update(keys[-iraws_window:])
        if len(keep) < len(keys):
            new_aux = _prune_aux_by_recency(new_aux, keep)
    return child._replace(aux_flat=new_aux)


# ============================================================================
# Model input with dummy subs (v5 variant)
# ============================================================================

def prepare_batched_input_v5_dummy(batch_data, device, max_subs=50,
                                    max_actions=900):
    """Batch model input.
    batch_data: list of (expr, resolved_subs, valid_actions, target_sector, target)

    Differences vs prepare_batched_input_v5:
    - No `subs` arg; dummy subs are RS keys with empty replacement dicts
      (variant F from v5_test_dummy_subs.py — verified |logit Δ|=1.5e-5,
      |prob Δ|=4.1e-7, top-1 always matches)
    - max_actions defaults to 900 (matches training distribution)
    """
    # MAX_REPLACEMENT_TERMS was previously imported from beam_search_full;
    # inlined here to keep v6 self-contained (beam_search_full now in archive/).
    MAX_REPLACEMENT_TERMS = 20  # Must match training

    batch_size = len(batch_data)
    max_terms = max(len(filter_to_sector(d[0], d[3])) for d in batch_data) if batch_data else 1
    max_terms = max(max_terms, 1)
    max_actions_eff = max(len(d[2]) for d in batch_data)
    max_actions_eff = min(max_actions_eff, max_actions)

    N = ibp_env.N_INDICES
    D = ibp_env.N_DENOMINATORS

    expr_integrals_np = np.zeros((batch_size, max_terms, N), dtype=np.int64)
    expr_coeffs_np = np.zeros((batch_size, max_terms), dtype=np.int64)
    expr_mask_np = np.zeros((batch_size, max_terms), dtype=bool)
    sub_keys_np = np.zeros((batch_size, max_subs, N), dtype=np.int64)
    sub_repl_ints_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS, N), dtype=np.int64)
    sub_repl_coeffs_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.int64)
    sub_repl_mask_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=bool)
    sub_mask_np = np.zeros((batch_size, max_subs), dtype=bool)
    action_ibp_ops_np = np.zeros((batch_size, max_actions_eff), dtype=np.int64)
    action_deltas_np = np.zeros((batch_size, max_actions_eff, N), dtype=np.int64)
    action_mask_np = np.zeros((batch_size, max_actions_eff), dtype=bool)
    sector_masks_np = np.zeros((batch_size, D), dtype=bool)
    target_integrals_np = np.zeros((batch_size, N), dtype=np.int64)

    for i, (expr, resolved_subs, valid_actions, target_sector, target) in enumerate(batch_data):
        sector_expr = filter_to_sector(expr, target_sector)
        expr_items = list(sector_expr.items())
        n_expr = len(expr_items)
        if n_expr > 0:
            expr_integrals_np[i, :n_expr] = [integral for integral, _ in expr_items]
            expr_coeffs_np[i, :n_expr] = [coeff for _, coeff in expr_items]
            expr_mask_np[i, :n_expr] = True

        # Dummy subs: RS keys (last 50), empty replacement → all-zero repl tensors
        rs_items = list(resolved_subs.keys())[-max_subs:]
        for j, key_integral in enumerate(rs_items):
            sub_mask_np[i, j] = True
            sub_keys_np[i, j] = key_integral
            # leave sub_repl_ints/coeffs/mask as zeros (verified equivalent)

        n_actions = min(len(valid_actions), max_actions_eff)
        if n_actions > 0:
            action_ibp_ops_np[i, :n_actions] = [op for op, _ in valid_actions[:n_actions]]
            action_deltas_np[i, :n_actions, :] = [delta for _, delta in valid_actions[:n_actions]]
            action_mask_np[i, :n_actions] = True

        sector_masks_np[i] = get_sector_mask(target)
        target_integrals_np[i] = target

    return {
        'expr_integrals': torch.from_numpy(expr_integrals_np).to(device),
        'expr_coeffs': torch.from_numpy(expr_coeffs_np).to(device),
        'expr_mask': torch.from_numpy(expr_mask_np).to(device),
        'sub_keys': torch.from_numpy(sub_keys_np).to(device),
        'sub_repl_ints': torch.from_numpy(sub_repl_ints_np).to(device),
        'sub_repl_coeffs': torch.from_numpy(sub_repl_coeffs_np).to(device),
        'sub_repl_mask': torch.from_numpy(sub_repl_mask_np).to(device),
        'sub_mask': torch.from_numpy(sub_mask_np).to(device),
        'action_ibp_ops': torch.from_numpy(action_ibp_ops_np).to(device),
        'action_deltas': torch.from_numpy(action_deltas_np).to(device),
        'action_mask': torch.from_numpy(action_mask_np).to(device),
        'sector_mask': torch.from_numpy(sector_masks_np).to(device),
        'target_integral': torch.from_numpy(target_integrals_np).to(device),
    }


# ============================================================================
# Beam search loop
# ============================================================================

def max_w12(expr, target_sector):
    nms = get_non_masters(expr, target_sector)
    if not nms:
        return (0, 0)
    return max((weight(k)[0], weight(k)[1]) for k in nms)


def total_w12(expr, target_sector):
    nms = get_non_masters(expr, target_sector)
    if not nms:
        return (0, 0)
    return (sum(weight(k)[0] for k in nms), sum(weight(k)[1] for k in nms))


def beam_search_v5(env, model, start_expr, target_sector, start_w12,
                   beam_width=40, max_steps=1000, device='cpu',
                   beam_sort='mixed', max_actions=900, ckpt_path=None,
                   ckpt_every=50, verbose=True,
                   resume_from=None, tabu=False,
                   use_incremental_aux=True,
                   use_exprkeyed=True,
                   ckpt_every_step=False,
                   iraws_window=None,
                   iraws_keep_first=None,
                   lazy_rs=True,
                   model_batch_chunk=8,
                   top_k=None):
    """Sequential beam search, v5 strip-active semantics.

    resume_from: path to a ckpt.pkl (or result.pkl) from a prior run; resumes
                 the beam from there at its recorded step.
    tabu: if True, maintain per-expr action tabu: never re-pick the same
          (target, ibp_op, delta) from the same expr fingerprint. Matches
          delta_beam_search.py's DELTA_TABU semantics.
    """
    resume_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        with open(resume_from, 'rb') as f:
            d = pickle.load(f)
            if isinstance(d, dict) and d.get('_streamed'):
                # streamed checkpoint: meta dict, then n_states state-dicts as
                # independent pickle frames (see _stream_dump_ckpt).
                _n = d['n_states']
                d = dict(d)
                d['beam'] = [pickle.load(f) for _ in range(_n)]
        resume_step = d['step']
        beam = []
        # Convert saved aux from picklable (op, seed)-keyed form back to id()-
        # keyed in-memory form. This preserves the EXACT iraws structure the
        # original run had (including incremental's (op, seed) dedup). The
        # alternative of force-rebuilding emits extra Phase-1b actions under
        # multiple anchors for the same (op, seed) raw, diverging from the
        # original trajectory.
        for s in d['beam']:
            if s.get('aux_flat') is not None:
                s['aux_flat'] = _aux_from_picklable(s['aux_flat'], env=env)
            # Recompute cached weight fields if missing
            if 'max_w12' not in s or 'total_w12' not in s:
                _nm = get_non_masters(s['expr'], target_sector)
                if _nm:
                    _wl = [(weight(k)[0], weight(k)[1]) for k in _nm]
                    s['max_w12'] = max(_wl)
                    s['total_w12'] = (sum(w[0] for w in _wl),
                                       sum(w[1] for w in _wl))
                else:
                    s['max_w12'] = (0, 0)
                    s['total_w12'] = (0, 0)
            beam.append(State_v5(**s))
        if verbose:
            print(f'[v6 RESUME] loaded {len(beam)} beam states from step '
                  f'{resume_step}', flush=True)
    else:
        initial_nm = get_non_masters(start_expr, target_sector)
        if initial_nm:
            _w_list = [(weight(k)[0], weight(k)[1]) for k in initial_nm]
            init_mw = max(_w_list)
            init_tw = (sum(w[0] for w in _w_list), sum(w[1] for w in _w_list))
        else:
            init_mw = (0, 0)
            init_tw = (0, 0)
        initial = State_v5(
            expr=dict(start_expr),
            resolved_subs={},
            sub_accum={},
            score=0.0,
            path=[],
            n_non_masters=len(initial_nm),
            max_w12=init_mw,
            total_w12=init_tw,
        )
        beam = [initial]
    best_state = beam[0]
    initial_mw = max_w12(start_expr, target_sector)
    t0 = time.time()

    # Tabu: expr fingerprint -> set of (target, ibp_op, delta) tried before
    tabu_dict = {} if tabu else None
    # Restore tabu from ckpt if present (resume bit-identical needs this).
    if (resume_from is not None and os.path.exists(resume_from)
            and tabu_dict is not None and d.get('tabu_dict')):
        saved_tabu = d['tabu_dict']
        # saved as list of (expr_items, [(target, op, delta), ...])
        for expr_items, entries in saved_tabu:
            tabu_dict[frozenset(expr_items)] = {
                (tuple(t), op, tuple(dlt)) for t, op, dlt in entries
            }
        if verbose:
            n_entries = sum(len(v) for v in tabu_dict.values())
            print(f'[v6 RESUME] restored tabu_dict: {len(tabu_dict)} expr '
                  f'fingerprints, {n_entries} total entries', flush=True)
    # Free the loaded checkpoint dict — beam + tabu have been extracted into
    # live State_v5 objects, so keeping `d` bound pins the ENTIRE serialized
    # beam (GBs of aux) alive for the whole run (diagnostic artifact + leak).
    if resume_from is not None and os.path.exists(resume_from):
        del d

    def sort_weight(s):
        return (s.max_w12, s.n_non_masters, -s.score)

    def sort_totalweight(s):
        return (s.total_w12, s.n_non_masters, -s.score)

    # Hoist memprobe config out of the per-step loop. Per-step we want at
    # most a set-membership check or an integer modulo — no env-var parsing.
    _memprobe_path = os.environ.get('SAILIR_MEMPROBE_FULL')
    _memprobe_steps_set = None
    _memprobe_every = None
    _memprobe_start = None
    if _memprobe_path:
        _steps_env = os.environ.get('SAILIR_MEMPROBE_FULL_STEPS')
        if _steps_env:
            _memprobe_steps_set = {int(x) for x in _steps_env.split(',') if x.strip()}
        else:
            _memprobe_every = int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_EVERY', '10'))
            _memprobe_start = int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_START', '20'))

    # Glibc memory tuning.
    #  (1) End-of-step malloc_trim(0) — DEFAULT ON. Returns the freed heap top
    #      to the OS; helped at shallow depth and is neutral at worst (it only
    #      ever reclaims the trimmable top). Opt out: SAILIR_END_OF_STEP_TRIM=0.
    #  (2) mmap-threshold pin via mallopt(M_MMAP_THRESHOLD=64MB) — DEFAULT OFF
    #      (OPT-IN: SAILIR_MMAP_THRESHOLD=<bytes>). It was in the validated
    #      probes but those only ran NORMAL integrals (74/84/LR); on the
    #      huge-equation HOG regime it BACKFIRES (+~3GB): forcing medium
    #      equation dicts onto the heap where live data blocks trim. So it is
    #      no longer applied unless explicitly requested.
    _end_of_step_trim = None
    try:
        import ctypes as _ctypes
        import ctypes.util as _ctypes_util
        _libc = _ctypes.CDLL(_ctypes_util.find_library('c'))
        _mmap_thr = os.environ.get('SAILIR_MMAP_THRESHOLD')
        if _mmap_thr:                                  # opt-in only
            _libc.mallopt.argtypes = [_ctypes.c_int, _ctypes.c_int]
            _libc.mallopt.restype = _ctypes.c_int
            _libc.mallopt(-3, int(_mmap_thr))          # M_MMAP_THRESHOLD = -3
        if os.environ.get('SAILIR_END_OF_STEP_TRIM', '1') != '0':
            _libc.malloc_trim.argtypes = [_ctypes.c_size_t]
            _libc.malloc_trim.restype = _ctypes.c_int
            _end_of_step_trim = _libc.malloc_trim
    except Exception:
        _end_of_step_trim = None

    # Optional beam-set trace: log per step (step, n_distinct_expr_fps,
    # hash_of_expr_fp_set) to verify the hypothesis "tabu trap fires only
    # when current beam's expr_fp set equals some prior step's set".
    # Gated by SAILIR_TRACE_BEAM_SETS=<path>. Per-step cost = one hash
    # over n_beam frozensets; per-step memory = 3 ints.
    _beam_set_trace_path = os.environ.get('SAILIR_TRACE_BEAM_SETS')
    _beam_set_history = []  # list of (step, n_distinct, set_hash)

    # Optional bounded-tabu mode (SAILIR_TABU_CAP=N, N>0). Per (expr_fp,target)
    # we accumulate every tried (op,delta) into a tabu set, exactly like the
    # baseline aggressive tabu, and pick top-K of valid minus that set. The ONE
    # difference: the effective block is capped at CAP actions. When the number
    # of currently-valid tabu'd actions exceeds CAP, we re-allow the highest-
    # model-prob overflow (keeping the lowest-prob CAP blocked) — so once we've
    # swept CAP-deep, the model's current favourites re-open and get re-tried
    # under the evolved RS (different sol → different children). Reduces to
    # baseline byte-for-byte until |valid∩tabu| first exceeds CAP. Replaces the
    # plain tabu filter — they're alternatives, not stackable. Storage grows
    # like baseline (the cap governs the FILTER, not the stored set).
    # Aggressive tabu is the FOUNDATION whenever --tabu is set: every top-K
    # pick is recorded and blocked (== baseline). SAILIR_TABU_CAP only adds the
    # optional cap+cycling LAYER on top: cap>0 caps the effective block at CAP
    # and re-allows the highest-prob overflow once swept CAP-deep; cap<=0 means
    # NO cap (effective block = ∞), i.e. pure aggressive baseline tabu.
    _tabu_cap = int(os.environ.get('SAILIR_TABU_CAP', '0'))
    _tabu_cap_eff = _tabu_cap if _tabu_cap > 0 else float('inf')
    _bounded_tabu_on = bool(tabu)
    _bt_tabu = {} if _bounded_tabu_on else None

    # Focused runtime memory breakdown (SAILIR_MEM_BREAKDOWN=1). Each sampled
    # step we RESET VmHWM at step start (/proc/self/clear_refs) so the print's
    # peak_rss is THIS step's transient peak (not the all-time max), then report:
    #   peak_rss   = per-step transient peak (pre end-of-step trim)
    #   rss        = post-trim quiescent RSS
    #   smaps      = where RSS lives: heap (brk) / anon (mmap) / file
    #   named structs (shared-seen partition): beam,cand,cand_meta,cache,bt,tabu
    #   glibc_transient = peak_rss - rss  (freed-then-trimmed each step)
    #   live_unaccounted = rss - sum_struct  (live, unnamed -> next target)
    # Every 4th sample a gc deep-walk-by-type names any live_unaccounted.
    _mem_bd = os.environ.get('SAILIR_MEM_BREAKDOWN')
    _mem_bd_every = int(os.environ.get('SAILIR_MEM_BREAKDOWN_EVERY', '25'))

    def _membd_deepsize(o, seen=None):
        import sys as _s
        from collections import deque as _dq
        if seen is None:
            seen = set()
        tot = 0; st = _dq([o])
        while st:
            x = st.popleft(); i = id(x)
            if i in seen:
                continue
            seen.add(i)
            if isinstance(x, np.ndarray):
                tot += x.nbytes; continue
            tot += _s.getsizeof(x, 0)
            if isinstance(x, dict):
                for k, v in x.items():
                    st.append(k); st.append(v)
            elif isinstance(x, (list, tuple, set, frozenset, _dq)):
                for e in x:
                    st.append(e)
            else:
                # arbitrary object: traverse its attributes so env/model/caches
                # held in __dict__ or __slots__ are actually counted.
                _xd = getattr(x, '__dict__', None)
                if _xd is not None:
                    st.append(_xd)
                _sl = getattr(type(x), '__slots__', None)
                if _sl:
                    for _n in _sl:
                        try:
                            st.append(getattr(x, _n))
                        except Exception:
                            pass
        return tot

    def _membd_mem():
        rss = hwm = 0
        with open('/proc/self/status') as _f:
            for _l in _f:
                if _l.startswith('VmRSS:'):
                    rss = int(_l.split()[1])
                elif _l.startswith('VmHWM:'):
                    hwm = int(_l.split()[1])
        return rss, hwm   # kB; hwm here is the per-step peak after a reset

    def _membd_reset_peak():
        # Reset VmHWM to current VmRSS so the next VmHWM read is THIS step's
        # peak. CLEAR_REFS_MM_HIWATER_RSS = '5'. Cheap; no side effects beyond
        # the peak counter (which only the diagnostic relies on).
        try:
            with open('/proc/self/clear_refs', 'w') as _cf:
                _cf.write('5')
        except Exception:
            pass

    def _membd_smaps():
        # Sum Rss (kB) by region category from /proc/self/smaps:
        #   heap = the main-arena brk region ([heap]); anon = anonymous mmaps
        #   (glibc secondary arenas + large malloc/numpy/torch); file = mapped
        #   files (libs, model). Tells WHERE the RSS physically lives.
        heap = anon = filed = 0
        is_heap = is_file = False
        try:
            with open('/proc/self/smaps') as _f:
                for _l in _f:
                    # Mapping header lines start with "start-end" (a '-' in the
                    # first token); detail lines are "Field:  N kB".
                    if '-' in _l.split(' ', 1)[0]:
                        _p = _l.split()
                        _path = _p[5] if len(_p) >= 6 else ''
                        is_heap = (_path == '[heap]')
                        is_file = _path.startswith('/')
                    elif _l.startswith('Rss:'):
                        _kb = int(_l.split()[1])
                        if is_heap:
                            heap += _kb
                        elif is_file:
                            filed += _kb
                        else:
                            anon += _kb
        except Exception:
            pass
        return heap, anon, filed

    def _membd_torch_bytes():
        # Sum unique LIVE torch tensor storage (model params + forward
        # transients: probs_all, batch_data, ...). These live in anon-mmap,
        # OUTSIDE the Python heap, so _membd_deepsize can't see them — they are
        # the dominant PEAK driver on heavy candidate-gen steps.
        try:
            import torch as _t
            import gc as _g
            _seen_st = set()
            _tot = 0
            for _o in _g.get_objects():
                if isinstance(_o, _t.Tensor):
                    try:
                        _st = _o.untyped_storage()
                        _sid = id(_st)
                        if _sid not in _seen_st:
                            _seen_st.add(_sid)
                            _tot += _st.nbytes()
                    except Exception:
                        pass
            return _tot
        except Exception:
            return 0

    def _membd_mallinfo():
        # glibc allocator stats via malloc_info() XML (mallinfo2 doesn't exist
        # on glibc 2.28; old mallinfo() int fields overflow past 2GB). Reports
        # glibc-managed memory ONLY (large allocs / numpy / torch mmap; NOT
        # CPython pymalloc small-object arenas — those are counted by the gc
        # heap walk instead). free = glibc freed-but-retained (fragmentation);
        # mmap = bytes in mmap'd regions; system = total from-OS in glibc arenas.
        try:
            import ctypes as _c
            import re as _re
            _libc2 = _c.CDLL('libc.so.6')
            _buf = _c.c_char_p()
            _sz = _c.c_size_t()
            _libc2.open_memstream.restype = _c.c_void_p
            _libc2.open_memstream.argtypes = [
                _c.POINTER(_c.c_char_p), _c.POINTER(_c.c_size_t)]
            _fp = _libc2.open_memstream(_c.byref(_buf), _c.byref(_sz))
            _libc2.malloc_info.argtypes = [_c.c_int, _c.c_void_p]
            _libc2.malloc_info(0, _fp)
            _libc2.fflush(_c.c_void_p(_fp))
            _libc2.fclose(_c.c_void_p(_fp))
            _xml = _c.string_at(_buf.value, _sz.value).decode()
            _libc2.free.argtypes = [_c.c_void_p]
            _libc2.free(_c.cast(_buf, _c.c_void_p))

            def _last(_pat):
                _mm = _re.findall(_pat, _xml)
                return int(_mm[-1]) if _mm else 0
            _fast = _last(r'<total type="fast" count="\d+" size="(\d+)"')
            _rest = _last(r'<total type="rest" count="\d+" size="(\d+)"')
            _mmap = _last(r'<total type="mmap" count="\d+" size="(\d+)"')
            _syscur = _last(r'<system type="current" size="(\d+)"')
            _free = _fast + _rest
            return {'system': _syscur, 'hblkhd': _mmap, 'fordblks': _free,
                    'uordblks': max(_syscur - _free, 0)}
        except Exception:
            return None

    def _membd_gc_by_type(top_n=15):
        # FULL live-heap walk. CPython UNTRACKS dicts/tuples/lists whose contents
        # are all atomic (int/str) — gc.get_objects() does NOT list them, and
        # SAILIR's equation dicts (seed→int-coeff) are exactly that. So we seed
        # from the GC roots and traverse gc.get_referents(), which DOES reach
        # untracked children. id-dedup; numpy/torch buffers special-cased. This
        # reaches every reachable live object → total = the real python RSS, and
        # the by-type rows name the dominant container. Returns
        # (total_bytes, [(typename, bytes), ...]).
        import gc as _g
        import sys as _s
        from collections import Counter as _Cn
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import torch as _tt
        except Exception:
            _tt = None
        _g.collect()
        _seen = set()
        _tot = 0
        _bt = _Cn()
        _st_seen = set()
        _stack = _g.get_objects()                 # tracked roots
        _get_ref = _g.get_referents
        # Exclude the walk's OWN structures (seen/stack/bt/st_seen) so we never
        # count the instrumentation as heap (the seen-set alone is GBs on a big
        # heap — that observer effect inflated the total above rss).
        _self_ids = {id(_seen), id(_bt), id(_st_seen), id(_stack)}
        while _stack:
            _o = _stack.pop()
            _i = id(_o)
            if _i in _seen:
                continue
            _seen.add(_i)
            if _i in _self_ids:
                continue
            if _np is not None and isinstance(_o, _np.ndarray):
                _sz = _o.nbytes
            elif _tt is not None and isinstance(_o, _tt.Tensor):
                try:
                    _sid = id(_o.untyped_storage())
                    if _sid in _st_seen:
                        _sz = _s.getsizeof(_o, 0)
                    else:
                        _st_seen.add(_sid)
                        _sz = _o.untyped_storage().nbytes()
                except Exception:
                    _sz = _s.getsizeof(_o, 0)
            elif (_tt is not None and 'Storage' in type(_o).__name__
                  and hasattr(_o, 'nbytes')):
                # Storage object reached directly. getsizeof(storage) returns the
                # BUFFER size, so without this branch torch storage is counted
                # twice (once via its Tensor above). Dedup by storage id across
                # both paths so each buffer is counted exactly once.
                _sid = id(_o)
                if _sid in _st_seen:
                    _sz = 0
                else:
                    _st_seen.add(_sid)
                    try:
                        _sz = _o.nbytes()
                    except Exception:
                        _sz = _s.getsizeof(_o, 0)
            else:
                _sz = _s.getsizeof(_o, 0)
            _tot += _sz
            _typ = type(_o)
            _modn = getattr(_typ, '__module__', '')
            if not isinstance(_modn, str):
                _modn = ''
            _nm = getattr(_typ, '__name__', None)
            if not isinstance(_nm, str):
                _nm = repr(_typ)
            _bt[_modn + '.' + _nm] += _sz
            # Follow referents to reach UNTRACKED children (the equation dicts).
            for _r in _get_ref(_o):
                if id(_r) not in _seen:
                    _stack.append(_r)
        return _tot, _bt.most_common(top_n)

    # Diagnostic: dump per-task (expr_fp, target, n_v, picked_actions, V) every
    # step to <dir>/picks_step<N>.pkl. Used to diff baseline-tabu vs rank-cycle
    # picks and locate the first divergence. Off unless SAILIR_PICK_DUMP set.
    _pick_dump_dir = os.environ.get('SAILIR_PICK_DUMP')

    # Optional tabu audit / sol fallback infra.
    # SAILIR_TABU_AUDIT=1       → maintain _sol_history + dump audit on stuck
    # SAILIR_TABU_SOL_FALLBACK=1 → maintain _sol_history AND consult it on
    #                              stuck to let "drifted" actions through.
    # Either flag activates _sol_history. No cap on fallback uses — it only
    # fires when primary tabu has zero valid actions, so it's self-limiting.
    _tabu_audit_on = os.environ.get('SAILIR_TABU_AUDIT') == '1'
    _tabu_sol_fallback = os.environ.get('SAILIR_TABU_SOL_FALLBACK') == '1'
    # _sol_history[frozenset(expr_fp)][(target, op, delta)] = list of (step, sol_fp)
    _sol_history = ({} if (_tabu_audit_on or _tabu_sol_fallback) else None)
    _sol_fallback_count = 0

    for step in range(resume_step, max_steps):
        # Termination: ANY beam state has drained — we only need one successful
        # path (the proof). Don't wait for every beam slot to finish.
        winning = next((s for s in beam if s.n_non_masters == 0), None)
        if winning is not None:
            best_state = winning
            if verbose:
                print(f'[v6 step {step}] beam state drained — DONE '
                      f'(path_len={len(winning.path)})', flush=True)
            break

        # Reset the peak-RSS counter at the START of a sampled step so the
        # MEMBD print reads THIS step's transient peak (the candidate-gen
        # spike), not the all-time high-water mark.
        if _mem_bd and (step + 1) % _mem_bd_every == 0:
            _membd_reset_peak()

        # Record per-step beam set for the trap-hypothesis verification.
        if _beam_set_trace_path is not None:
            _bs = frozenset(frozenset(s.expr.items()) for s in beam)
            _beam_set_history.append((step, len(_bs), hash(_bs)))

        # ── v6: macro-beam dedup ──────────────────────────────────────────
        # Collapse beam slots that share the same expr_fp. The model only
        # sees expr + RS_KEYS (not values; see prepare_batched_input_v5_dummy)
        # so duplicate-expr states return the same model scores → waste.
        # Keep best variant per expr_fp by progress key (low max_w12, low nm,
        # high score).
        _dedup_t = time.time()
        macro_groups = {}  # expr_fp → best State
        for s in beam:
            efp = frozenset(s.expr.items())
            cur = macro_groups.get(efp)
            if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                              (cur.max_w12, cur.n_non_masters, -cur.score):
                macro_groups[efp] = s
        n_orig_beam = len(beam)
        beam = list(macro_groups.values())
        n_macros = len(beam)
        # Trace this collapse — it's the v6 telemetry the user cares about.
        if verbose and n_macros < n_orig_beam:
            print(f'[v6 step {step}] macro-dedup: '
                  f'{n_orig_beam} → {n_macros} distinct expr', flush=True)
        # ──────────────────────────────────────────────────────────────────

        # Build candidate list across all parents × valid actions
        candidates = []  # list of state_v5_child
        cand_metadata = []  # parallel list of (parent_state, target) for incremental aux
        all_state_summaries = []
        t_step = time.time()
        # V5_PROFILE=1: per-phase wall-clock instrumentation matching v4's
        # P1/P2/P3/P4 taxonomy from delta_beam_search.py.
        _v5_prof = bool(os.environ.get('V5_PROFILE'))
        # P1 = build tasks + valid actions: nm_tied + aux + gv + tabu
        # P2 = model batched scoring: batch_prep + model_fwd
        # P3 = apply children + beam selection: apply + sort
        # P4 = survivor materialize: attach_aux
        _p1 = {'nm_tied': 0.0, 'aux': 0.0, 'gv': 0.0, 'tabu': 0.0}
        _p2 = {'batch_prep': 0.0, 'model_fwd': 0.0}
        _p3 = {'apply': 0.0, 'sort': 0.0}
        _p4 = {'attach_aux': 0.0}
        _ct = {'aux_build_calls': 0, 'aux_repack_calls': 0,
               'enum_calls': 0, 'apply_calls': 0, 'attach_calls': 0,
               'n_iraws_total': 0, 'n_valid_total': 0,
               'cu_size_max': 0, 'rs_max': 0}

        # Per-state: compute indirect cache, enumerate tied targets, score actions
        # Build tasks for model batched inference
        tasks = []  # (parent_idx, target, valid)
        for parent_idx, s in enumerate(beam):
            if s.n_non_masters == 0:
                continue
            _t = time.time() if _v5_prof else 0
            nm = get_non_masters(s.expr, target_sector)
            mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
            tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
            if _v5_prof:
                _p1['nm_tied'] += time.time() - _t
            # Dummy subs (RS keys with empty values) — only used as keys for
            # enumerate_valid_actions iteration (it reads keys, not values).
            dummy_subs = {k: {} for k in s.resolved_subs.keys()}
            # Use cached aux_flat if available, else rebuild from scratch.
            _t = time.time() if _v5_prof else 0
            if s.aux_flat is not None:
                indirect_cache = _aux_to_result(s.aux_flat, env)
                if _v5_prof:
                    _ct['aux_repack_calls'] += 1
            else:
                # v7: depth-keyed from-scratch rebuild (no exprkeyed), then pack
                # the cu for storage. The returned indirect_cache stays dict
                # (consumed transiently by enumerate); only new_aux is packed.
                if (iraws_window is not None or iraws_keep_first is not None):
                    keys = list(s.resolved_subs.keys())
                    keep = set()
                    if iraws_keep_first is not None:
                        keep.update(keys[:iraws_keep_first])
                    if iraws_window is not None:
                        keep.update(keys[-iraws_window:])
                    if len(keep) < len(keys):
                        keep_list = [k for k in keys if k in keep]
                        fresh_dummy = {k: {} for k in keep_list}
                    else:
                        fresh_dummy = dummy_subs
                else:
                    fresh_dummy = dummy_subs
                indirect_cache, new_aux = compute_indirect_substituted_with_aux(
                    fresh_dummy, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts,
                    env._raw_eq_cache,
                )
                new_aux = _v7_aux_pack(new_aux)  # dict cu -> PackedEq for storage
                if _v5_prof:
                    _ct['aux_build_calls'] += 1
                beam[parent_idx] = s._replace(aux_flat=new_aux)
                s = beam[parent_idx]
            if _v5_prof:
                _p1['aux'] += time.time() - _t
            if _v5_prof:
                _ct['n_iraws_total'] += len(indirect_cache)
                if s.aux_flat:
                    _ct['cu_size_max'] = max(_ct['cu_size_max'],
                                              len(s.aux_flat[0]))
                _ct['rs_max'] = max(_ct['rs_max'], len(s.resolved_subs))
            for target in tied:
                _t = time.time() if _v5_prof else 0
                valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                    target, indirect_cache, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts, 'subsector',
                    env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                )
                if _v5_prof:
                    _p1['gv'] += time.time() - _t
                    _ct['enum_calls'] += 1
                # Tabu: filter out (target, op, delta) tried previously from
                # this expr fingerprint.
                if (not _bounded_tabu_on
                        and tabu_dict is not None and valid):
                    _t = time.time() if _v5_prof else 0
                    expr_fp = frozenset(s.expr.items())
                    tabu_set = tabu_dict.get(expr_fp)
                    if tabu_set:
                        valid = [(op, dlt) for (op, dlt) in valid
                                 if (target, op, dlt) not in tabu_set]
                    if _v5_prof:
                        _p1['tabu'] += time.time() - _t
                # Aggressive tabu: remove this (expr_fp,target)'s tabu'd actions
                # here — BEFORE the max_actions truncation — so the model sees
                # the same tabu-free window as baseline (byte-identical). TWO
                # cases keep the FULL list instead, deferring to the pick loop:
                #  • EXHAUSTED (every valid action tabu'd): filtering would drop
                #    the task (the trap → stuck). Keep the full list so the pick
                #    loop CYCLES — re-picks the top-K best to retry under the
                #    evolved RS instead of dying.
                #  • OVER fixed cap (only when SAILIR_TABU_CAP>0): pick loop
                #    re-allows the highest-prob overflow using model probs.
                elif _bounded_tabu_on and valid:
                    _t = time.time() if _v5_prof else 0
                    _bt_key = (frozenset(s.expr.items()), target)
                    _bt_set = _bt_tabu.get(_bt_key)
                    if _bt_set:
                        _nblk = sum(1 for a in valid if a in _bt_set)
                        if _nblk < len(valid) and _nblk <= _tabu_cap_eff:
                            valid = [a for a in valid if a not in _bt_set]
                    if _v5_prof:
                        _p1['tabu'] += time.time() - _t
                if valid:
                    if _v5_prof:
                        _ct['n_valid_total'] += len(valid)
                    tasks.append((parent_idx, target, valid))

        # Instrumentation: dump tasks/valid lists at a specific step
        _dump_step = os.environ.get('V5_DUMP_VALIDS_AT_STEP')
        if _dump_step and int(_dump_step) == step + 1:
            _dump_path = os.environ.get('V5_DUMP_PATH',
                                        f'/tmp/v5_valids_step{step+1}.pkl')
            with open(_dump_path, 'wb') as _df:
                pickle.dump([
                    (pi, tuple(tg), tuple((op, tuple(d)) for op, d in v))
                    for pi, tg, v in tasks
                ], _df)
            print(f'  [DUMP] valids at step {step+1} → {_dump_path}', flush=True)

        if not tasks:
            # SOL_FALLBACK: before declaring STUCK, retry filter using
            # sol_fp drift. For each raw candidate (op, dlt), compute the
            # current sol_fp under THIS state's RS. If it doesn't match
            # ANY historical sol_fp under (target, op, dlt) for this
            # expr_fp, that means the substitution we'd apply now is
            # different from what was tabu'd → allow.
            if _tabu_sol_fallback and _sol_history is not None:
                tasks = []
                n_unblocked_total = 0
                for parent_idx in range(len(beam)):
                    s = beam[parent_idx]
                    nm = get_non_masters(s.expr, target_sector)
                    if not nm:
                        continue
                    mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
                    tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
                    expr_fp_ = frozenset(s.expr.items())
                    ts_ = (tabu_dict.get(expr_fp_)
                           if tabu_dict is not None else None)
                    hist_p = _sol_history.get(expr_fp_)
                    if hist_p is None:
                        continue
                    ic = (_aux_to_result(s.aux_flat, env)
                          if s.aux_flat is not None else None)
                    if ic is None:
                        continue
                    dummy_subs = {k: {} for k in s.resolved_subs.keys()}
                    cu_p = s.aux_flat[0]
                    rid_p = s.aux_flat[2]
                    N_ = ibp_env.N_INDICES
                    for tgt in tied:
                        raw_valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                            tgt, ic, s.resolved_subs,
                            env.ibp_t, env.li_t, env.shifts, 'subsector',
                            env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                        )
                        unblocked = []
                        for (op, dlt) in raw_valid:
                            # If primary tabu doesn't block it, keep as-is.
                            if ts_ is None or (tgt, op, dlt) not in ts_:
                                unblocked.append((op, dlt))
                                continue
                            # Primary blocks it — check sol_fp drift.
                            key_ = (tgt, op, dlt)
                            hist = hist_p.get(key_, [])
                            if not hist:
                                continue  # primary tabu'd via different path
                            seed_p = tuple(tgt[i] + dlt[i] for i in range(N_))
                            cu_idx_p = rid_p.get((op, seed_p))
                            if cu_idx_p is None:
                                continue
                            sol_p = solve_ibp_for(_v7_as_dict(cu_p[cu_idx_p]), tgt)
                            if sol_p is None:
                                continue
                            cur_sol_fp = hash(frozenset(sol_p.items()))
                            hist_sol_fps = {sfp for _s, sfp in hist}
                            if cur_sol_fp not in hist_sol_fps:
                                # Drifted: substitution differs from
                                # what was previously tabu'd → allow.
                                unblocked.append((op, dlt))
                        if unblocked:
                            tasks.append((parent_idx, tgt, unblocked))
                            n_unblocked_total += len(unblocked)
                if tasks:
                    _sol_fallback_count += 1
                    if verbose:
                        print(f'[v6 step {step}] TABU TRAP - sol_fp fallback '
                              f'#{_sol_fallback_count}: '
                              f'{n_unblocked_total} drifted actions unblocked',
                              flush=True)
                    # Fall through to model forward / action selection.
                else:
                    if verbose:
                        print(f'[v6 step {step}] no tasks — STUCK (sol_fp '
                              f'fallback found no drifted actions)', flush=True)
            if not tasks:
                if verbose:
                    print(f'[v6 step {step}] no tasks — STUCK', flush=True)
                # All the diagnostics below only fire if STILL stuck after
                # the sol_fp fallback above (or if fallback was disabled).
                _real_stuck = True
            else:
                _real_stuck = False
            # If beam-set tracing is on, identify prior steps whose beam
            # had the SAME set of expr_fps as the current (stuck) beam.
            if _real_stuck and _beam_set_trace_path is not None:
                _cur_bs = frozenset(frozenset(s.expr.items()) for s in beam)
                _cur_h = hash(_cur_bs)
                _prior_matches = [s for s, _n, h in _beam_set_history
                                  if h == _cur_h and s < step]
                with open(_beam_set_trace_path, 'w') as _tf:
                    _tf.write(f'stuck_step\t{step}\n')
                    _tf.write(f'stuck_n_distinct\t{len(_cur_bs)}\n')
                    _tf.write(f'prior_steps_with_same_set\t'
                               f'{",".join(map(str, _prior_matches))}\n')
                    _tf.write(f'# history (step, n_distinct, hash):\n')
                    for s, n, h in _beam_set_history:
                        _tf.write(f'{s}\t{n}\t{h}\n')
                print(f'  [BEAM_SET_TRACE] stuck-set seen earlier at steps: '
                      f'{_prior_matches} → '
                      f'{_beam_set_trace_path}', flush=True)
            # Comprehensive forensic dump of the stuck state. Gated by env
            # var SAILIR_DUMP_STUCK=<path> so it only fires when requested.
            _dump_stuck_path = (os.environ.get('SAILIR_DUMP_STUCK')
                                 if _real_stuck else None)
            if _dump_stuck_path:
                try:
                    _stuck_payload = {
                        'step': step,
                        'target_sector': target_sector,
                        'start_w12': start_w12,
                        'beam_width': beam_width,
                        'max_actions': max_actions,
                        # Tabu — every (target, op, delta) recorded
                        'tabu_dict': {
                            tuple(expr_fp): list(entries)
                            for expr_fp, entries in tabu_dict.items()
                        } if tabu_dict is not None else None,
                        'tabu_summary': ({
                            'n_expr_fps': len(tabu_dict),
                            'total_entries': sum(len(v) for v in tabu_dict.values()),
                        } if tabu_dict is not None else None),
                        # Full beam: every state's expr, RS, sub_accum, path,
                        # score, max_w12, total_w12, n_non_masters, aux_flat
                        'beam': [
                            {
                                'expr': dict(st.expr),
                                'resolved_subs': (dict(st.resolved_subs)
                                                   if st.resolved_subs else None),
                                'sub_accum': (dict(st.sub_accum)
                                              if st.sub_accum else None),
                                'path': list(st.path) if st.path else [],
                                'score': float(st.score),
                                'max_w12': tuple(st.max_w12),
                                'total_w12': tuple(st.total_w12),
                                'n_non_masters': int(st.n_non_masters),
                                'aux_flat': (_aux_to_picklable(st.aux_flat)
                                              if st.aux_flat is not None else None),
                            }
                            for st in beam
                        ],
                        # Per-state, per-target: what enumerate_valid_actions
                        # gave us (raw) and what tabu LEFT (post-filter), so
                        # we can see EXACTLY how many candidates each target
                        # had vs how many tabu took out.
                        'per_state_per_target': [],
                        # Best state tracker
                        'best_state': ({
                            'path_len': int(best_state.path_len if hasattr(best_state, 'path_len') else len(best_state.path)),
                            'n_non_masters': int(best_state.n_non_masters),
                            'max_w12': tuple(best_state.max_w12),
                            'score': float(best_state.score),
                        } if best_state is not None else None),
                    }
                    # Re-enumerate per parent, per tied target — capture
                    # raw_valid (no tabu) and filtered_valid (post-tabu).
                    for parent_idx, st in enumerate(beam):
                        nm = get_non_masters(st.expr, target_sector)
                        if not nm:
                            continue
                        mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
                        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
                        # rebuild indirect_cache for re-enumeration
                        ic = (_aux_to_result(st.aux_flat, env)
                              if st.aux_flat is not None else None)
                        if ic is None:
                            continue
                        dummy_subs = {k: {} for k in st.resolved_subs.keys()}
                        expr_fp_ = frozenset(st.expr.items())
                        ts_ = (tabu_dict.get(expr_fp_)
                               if tabu_dict is not None else None)
                        cu_p = st.aux_flat[0]
                        rid_p = st.aux_flat[2]
                        N_ = ibp_env.N_INDICES
                        _hist_p = (_sol_history.get(expr_fp_)
                                    if _sol_history is not None else None)
                        for tgt in tied:
                            raw_valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                                tgt, ic, st.resolved_subs,
                                env.ibp_t, env.li_t, env.shifts, 'subsector',
                                env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                            )
                            filtered = ([(op, dlt) for (op, dlt) in raw_valid
                                          if (tgt, op, dlt) not in ts_]
                                        if ts_ else list(raw_valid))
                            # AUDIT: for each tabu'd action, compute CURRENT
                            # sol_fp under THIS step's RS and compare to the
                            # sol_fp(s) when it was first tabu'd. Drift =
                            # current ≠ all historical sol_fps for that key.
                            audit_rows = []
                            if _hist_p is not None:
                                for (op, dlt) in raw_valid:
                                    key_ = (tgt, op, dlt)
                                    hist = _hist_p.get(key_, [])
                                    seed_p = tuple(tgt[i] + dlt[i] for i in range(N_))
                                    cu_idx_p = rid_p.get((op, seed_p))
                                    cur_sol_fp = None
                                    if cu_idx_p is not None:
                                        sol_p = solve_ibp_for(_v7_as_dict(cu_p[cu_idx_p]), tgt)
                                        if sol_p is not None:
                                            cur_sol_fp = hash(frozenset(sol_p.items()))
                                    hist_sol_fps = {sfp for _s, sfp in hist}
                                    drifted = (cur_sol_fp is not None
                                                and cur_sol_fp not in hist_sol_fps
                                                and len(hist) > 0)
                                    audit_rows.append({
                                        'op': op, 'delta': tuple(dlt),
                                        'current_sol_fp': cur_sol_fp,
                                        'history': hist,  # list of (step, sol_fp)
                                        'drifted': drifted,
                                    })
                            _stuck_payload['per_state_per_target'].append({
                                'parent_idx': parent_idx,
                                'target': tgt,
                                'n_raw': len(raw_valid),
                                'n_after_tabu': len(filtered),
                                'raw_valid': [(op, tuple(d)) for op, d in raw_valid],
                                'filtered': [(op, tuple(d)) for op, d in filtered],
                                'audit': audit_rows,
                            })
                    with open(_dump_stuck_path, 'wb') as _df:
                        pickle.dump(_stuck_payload, _df)
                    print(f'  [STUCK_DUMP] wrote forensic state → '
                          f'{_dump_stuck_path}', flush=True)
                except Exception as _e:
                    print(f'  [STUCK_DUMP] failed: {_e}', flush=True)
            if _real_stuck:
                break

        # Run model batched on all tasks (optionally chunked).
        # Chunking bounds the peak transient activation memory in the forward
        # pass — large unchunked batches leave hundreds of MB in glibc's free
        # list every step (the activations are freed but glibc retains pages).
        # Per-chunk prepare + forward + write into a preallocated probs_all,
        # with explicit `del` so each chunk's tensors are released before the
        # next. None / 0 / >= len(batch_data) means no chunking (original).
        _t = time.time() if _v5_prof else 0
        batch_data = []
        for parent_idx, target, valid in tasks:
            s = beam[parent_idx]
            batch_data.append((s.expr, s.resolved_subs, valid, target_sector, target))
        # Compute the global action width once so probs_all shape is stable.
        global_max_actions_eff = min(
            max(len(d[2]) for d in batch_data), max_actions)
        chunk_sz = model_batch_chunk if model_batch_chunk else len(batch_data)
        chunk_sz = max(1, min(chunk_sz, len(batch_data)))
        if _v5_prof:
            _p2['batch_prep'] += time.time() - _t
            _t = time.time()
        probs_all = torch.zeros(len(batch_data), global_max_actions_eff)
        for chunk_start in range(0, len(batch_data), chunk_sz):
            chunk = batch_data[chunk_start:chunk_start + chunk_sz]
            # Cap chunk's max_actions to the global so the slice assign below
            # has consistent width.
            b = prepare_batched_input_v5_dummy(chunk, device,
                                                max_actions=global_max_actions_eff)
            with torch.no_grad():
                _, chunk_probs = model(
                    b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                    b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                    b['sub_repl_mask'], b['sub_mask'],
                    b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                    b['sector_mask'], b['target_integral'],
                )
            cn = chunk_probs.shape[1]
            probs_all[chunk_start:chunk_start + len(chunk), :cn] = chunk_probs
            del b, chunk_probs
        probs = probs_all
        if _v5_prof:
            _p2['model_fwd'] += time.time() - _t

        # For each task, take top-K actions by prob, apply, generate candidates
        K = max(1, top_k if top_k is not None else beam_width // 2)
        _pick_dump_step = [] if _pick_dump_dir else None
        for ti, (parent_idx, target, valid) in enumerate(tasks):
            n_v = min(len(valid), probs.shape[1])
            row = probs[ti, :n_v].numpy()
            parent_state = beam[parent_idx]
            if _bounded_tabu_on:
                # Bounded tabu: block previously-tried (op,delta) for this
                # (expr_fp,target), but cap the effective block at CAP. When
                # more than CAP currently-valid actions are tabu'd, keep the
                # lowest-prob CAP blocked and RE-ALLOW the highest-prob overflow
                # (the model's current favourites get re-tried under evolved RS).
                _key_bt = (frozenset(parent_state.expr.items()), target)
                _tset = _bt_tabu.get(_key_bt)
                if _tset:
                    _blk = [i for i in range(n_v) if valid[i] in _tset]
                else:
                    _blk = []
                _n_blocked = len(_blk)
                if _n_blocked > _tabu_cap_eff:
                    _blk.sort(key=lambda i: row[i])      # ascending prob
                    _eff_block = set(_blk[:_tabu_cap])   # lowest-prob CAP
                else:
                    _eff_block = set(_blk)
                if _eff_block:
                    _avail = [i for i in np.argsort(-row) if i not in _eff_block]
                    if _avail:
                        top_idx = np.array(_avail[:K], dtype=int)
                    else:
                        # EXHAUSTED: every valid action is tabu'd. Cycle —
                        # re-allow everything and re-pick the top-K best, so a
                        # trapped task retries its best actions under the evolved
                        # RS rather than producing nothing (= baseline's stuck).
                        top_idx = np.argsort(-row)[:K]
                else:
                    top_idx = np.argsort(-row)[:K]
                # Record ALL picks into the tabu set (aggressive, like baseline).
                _ts2 = _bt_tabu.setdefault(_key_bt, set())
                for _ai in top_idx:
                    _ts2.add(valid[int(_ai)])
            else:
                # Take top-K by prob
                top_idx = np.argsort(-row)[:K]
                _n_blocked = -1
            if _pick_dump_step is not None:
                _V_d = _n_blocked
                _picked_d = tuple(sorted(
                    (op_, tuple(d_)) for (op_, d_)
                    in (valid[int(_a)] for _a in top_idx)))
                _pick_dump_step.append((
                    tuple(sorted(parent_state.expr.items())),
                    tuple(target), int(n_v), _picked_d, int(_V_d)))
            # Tabu is recorded LATER — after beam selection and macro-dedup
            # — for ONLY the actions that actually survived into the next
            # beam, not every top-K attempt.
            for ai in top_idx:
                ai = int(ai)
                op, delta = valid[ai]
                _t = time.time() if _v5_prof else 0
                result = apply_action_v5(
                    parent_state, target, op, delta, float(row[ai]),
                    env, target_sector, start_w12,
                    use_incremental_aux=use_incremental_aux,
                    lazy_rs=lazy_rs,
                )
                if _v5_prof:
                    _p3['apply'] += time.time() - _t
                    _ct['apply_calls'] += 1
                if result is None:
                    continue
                child, sol = result
                candidates.append(child)
                cand_metadata.append((parent_state, target, sol))

        if _pick_dump_step is not None:
            with open(f'{_pick_dump_dir}/picks_step{step}.pkl', 'wb') as _pf:
                pickle.dump(_pick_dump_step, _pf)

        if not candidates:
            if verbose:
                print(f'[v6 step {step}] no successful candidates — STUCK', flush=True)
            break

        # Beam selection
        _t_sort = time.time() if _v5_prof else 0
        # Build id() → metadata-index map so we can find each survivor's parent.
        meta_by_id = {id(c): i for i, c in enumerate(candidates)}
        if beam_sort == 'weight':
            candidates.sort(key=sort_weight)
            beam = candidates[:beam_width]
        elif beam_sort == 'mixed':
            half = beam_width // 2
            by_w = sorted(candidates, key=sort_weight)
            by_t = sorted(candidates, key=sort_totalweight)
            new_beam = []
            seen = set()
            for c in by_w:
                if len(new_beam) >= half:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            for c in by_t:
                if len(new_beam) >= beam_width:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            for c in by_w:
                if len(new_beam) >= beam_width:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            beam = new_beam
        else:
            raise ValueError(f'unknown beam_sort {beam_sort}')

        if _v5_prof:
            _p3['sort'] += time.time() - _t_sort

        # ── v6 early macro-dedup (Issue #11) ────────────────────────────
        # The original macro-dedup runs at the TOP of the next step. By
        # then we've already paid the cost of materializing aux_flat for
        # every duplicate child that's about to be thrown away. Doing the
        # dedup BEFORE materialization is bit-identical (dedup key depends
        # only on expr / max_w12 / n_non_masters / score — all of which
        # are unchanged by materialization) and saves the wasted work.
        # The next-step dedup becomes a no-op.
        if len(beam) > 1:
            early_groups = {}
            for s in beam:
                efp = frozenset(s.expr.items())
                cur = early_groups.get(efp)
                if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                                  (cur.max_w12, cur.n_non_masters, -cur.score):
                    early_groups[efp] = s
            if len(early_groups) < len(beam):
                if verbose:
                    print(f'[v6 step {step}] early-dedup: {len(beam)} → '
                          f'{len(early_groups)} distinct expr (pre-materialize)',
                          flush=True)
                beam = list(early_groups.values())

        # Tabu: now that beam survivors are finalised, record (target, op,
        # delta) of EACH survivor's last action under its parent's expr_fp.
        # Only survivors get tabu'd — top-K attempts whose children were
        # rejected by beam selection are NOT tabu'd (they may produce
        # different outcomes in future visits with different RS context).
        if tabu_dict is not None and not _bounded_tabu_on:
            N_ = ibp_env.N_INDICES
            for c in beam:
                meta_i = meta_by_id.get(id(c))
                if meta_i is None:
                    continue
                parent_state, target, _sol = cand_metadata[meta_i]
                # Last action in child's path = (target, op, delta) applied
                target_p, op_p, delta_p = c.path[-1]
                expr_fp_p = frozenset(parent_state.expr.items())
                ts = tabu_dict.setdefault(expr_fp_p, set())
                ts.add((target_p, op_p, delta_p))
                if _sol_history is not None:
                    seed_p = tuple(target_p[i] + delta_p[i] for i in range(N_))
                    cu_p_ = parent_state.aux_flat[0]
                    rid_p_ = parent_state.aux_flat[2]
                    cu_idx_p = rid_p_.get((op_p, seed_p))
                    if cu_idx_p is not None:
                        sol_p = solve_ibp_for(_v7_as_dict(cu_p_[cu_idx_p]), target_p)
                        if sol_p is not None:
                            sol_fp_p = hash(frozenset(sol_p.items()))
                            _hist = _sol_history.setdefault(expr_fp_p, {})
                            _hist.setdefault((target_p, op_p, delta_p),
                                              []).append((step, sol_fp_p))

        # Post-selection survivors: materialize lazy_rs THEN attach aux in a
        # single pass so the original `c` is still in meta_by_id.
        _t_mat = time.time() if _v5_prof else 0
        _t_aux = 0.0
        for i, c in enumerate(beam):
            parent_state, target, sol = cand_metadata[meta_by_id[id(c)]]
            if lazy_rs:
                c = _materialize_lazy_rs(c, parent_state, target, sol, start_w12)
            if use_incremental_aux:
                _t1 = time.time() if _v5_prof else 0
                c = _attach_incremental_aux(
                    c, parent_state, target, env, target_sector,
                    use_exprkeyed=use_exprkeyed,
                    iraws_window=iraws_window,
                    iraws_keep_first=iraws_keep_first,
                )
                if _v5_prof:
                    _t_aux += time.time() - _t1
                    _ct['attach_calls'] += 1
            beam[i] = c
        if _v5_prof:
            _p4['mat_rs'] = _p4.get('mat_rs', 0.0) + (time.time() - _t_mat - _t_aux)
            _p4['attach_aux'] += _t_aux

        # Track best so far: use (max_w, n_non_masters) only — ignore score.
        # Score is cumulative negative log-prob, so the initial state always
        # has the "highest" score and would never be displaced by progress.
        def _progress_key(s):
            return (s.max_w12, s.n_non_masters,
                    -len(s.path))  # tiebreak: prefer longer path (more progress)

        best_in_beam = min(beam, key=_progress_key)
        if _progress_key(best_in_beam) < _progress_key(best_state):
            best_state = best_in_beam

        if verbose:
            mw_best = max_w12(best_in_beam.expr, target_sector)
            nm_best = best_in_beam.n_non_masters
            sz_rs = len(best_in_beam.resolved_subs)
            rs_vsz = sum(len(v) for v in best_in_beam.resolved_subs.values())
            # v6: count distinct exprs in beam to measure diversity gain.
            n_uniq_expr = len({frozenset(s.expr.items()) for s in beam})
            print(f'[v6 step {step+1:>3}] beam={len(beam)} '
                  f'uniq_expr={n_uniq_expr} '
                  f'best mw={mw_best} nm={nm_best} '
                  f'rs={sz_rs} rs_vsz={rs_vsz} '
                  f'expr={len(best_in_beam.expr)} '
                  f'cand={len(candidates)} '
                  f't_step={time.time()-t_step:.1f}s '
                  f't_total={time.time()-t0:.1f}s',
                  flush=True)
            if _v5_prof:
                p1_t = sum(_p1.values())
                p2_t = sum(_p2.values())
                p3_t = sum(_p3.values())
                p4_t = sum(_p4.values())
                total = p1_t + p2_t + p3_t + p4_t
                step_t = time.time() - t_step
                p1_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p1.items())
                p3_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p3.items())
                p2_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p2.items())
                print(f'  PROF step={step+1} t_step={step_t:.2f}s '
                      f'P1={p1_t:.2f}s P2={p2_t:.2f}s P3={p3_t:.2f}s '
                      f'P4={p4_t:.2f}s residual={step_t-total:.2f}s', flush=True)
                print(f'    P1: {p1_parts}', flush=True)
                print(f'    P2: {p2_parts}', flush=True)
                print(f'    P3: {p3_parts}', flush=True)
                print(f'    cts: ' + ' '.join(f'{k}={v}' for k, v in _ct.items()),
                      flush=True)
        # Explicit aux dump (SAILIR_DUMP_AUX_STEPS="10,25"): print the best
        # state's aux_flat = (cu, ubm, rid, iraws) — lengths, the per-cu-entry
        # term counts, and samples — so depth-keyed vs exprkeyed aux can be
        # compared directly. SEPARATE block (not inside the profiling guard).
        _dump_steps = os.environ.get('SAILIR_DUMP_AUX_STEPS')
        if _dump_steps and (step + 1) in {int(_x) for _x in _dump_steps.split(',') if _x}:
            _ax = best_in_beam.aux_flat
            if _ax is None:
                print(f'[AUXDUMP step {step+1}] best.aux_flat is None', flush=True)
            else:
                _cu, _ubm, _rid, _iraws = _ax
                _sizes = sorted((len(c) for c in _cu), reverse=True)
                print(f'[AUXDUMP step {step+1}] use_exprkeyed={use_exprkeyed} '
                      f'| len(cu)={len(_cu)} len(iraws)={len(_iraws)} '
                      f'len(rid)={len(_rid)} | cu #terms total={sum(_sizes)} '
                      f'max={_sizes[0] if _sizes else 0} top8={_sizes[:8]}',
                      flush=True)
                print(f'  iraws[:4] = {_iraws[:4]}', flush=True)
                if _cu:
                    _big = max(_cu, key=len)
                    print(f'  largest cu entry: {len(_big)} terms; first 4 items'
                          f' = {list(_big.items())[:4]}', flush=True)
                print(f'  rid[:3] = {list(_rid.items())[:3]}', flush=True)

        def _stream_dump_ckpt(path):
            # Stream the checkpoint ONE beam-state at a time so each state's
            # to_dict-materialized cu is freed before the next. memray showed
            # the whole-beam to_dict (was _serialize_beam_for_ckpt) at 5.2GB /
            # 59% of peak — the entire beam's packed cu materialized as dicts
            # simultaneously. Each state is an INDEPENDENT pickle frame (fresh
            # memo per pickle.dump) so `del sd` actually frees it; a single
            # shared Pickler would pin every object in its memo and save
            # nothing. Format: a meta dict (_streamed=True, n_states) then
            # n_states state-dicts in one file; the resume path detects
            # _streamed and reads n_states frames. Serialization-only change:
            # the live beam is untouched (s._asdict() + _aux_to_picklable both
            # create fresh objects), so trajectory/result stay bit-identical.
            with open(path, 'wb') as f:
                pickle.dump({
                    '_streamed': True,
                    'step': step + 1,
                    'n_states': len(beam),
                    'target_sector': target_sector,
                    'start_w12': start_w12,
                    'tabu_dict': _serialize_tabu_for_ckpt(),
                }, f)
                for s in beam:
                    sd = s._asdict()
                    if sd.get('aux_flat') is not None:
                        sd['aux_flat'] = _aux_to_picklable(sd['aux_flat'])
                    pickle.dump(sd, f)
                    del sd

        def _serialize_tabu_for_ckpt():
            # Convert {frozenset(expr_items): set((target, op, delta), ...)} to
            # a picklable list. frozenset+set don't have stable ordering on
            # disk but content is preserved; restoration rebuilds the same
            # structure (set membership is what matters for tabu lookup).
            if tabu_dict is None:
                return None
            return [
                (tuple(expr_fp), [(tuple(t), op, tuple(dlt))
                                  for t, op, dlt in entries])
                for expr_fp, entries in tabu_dict.items()
            ]

        # Full-memory probe (gated by SAILIR_MEMPROBE_FULL=/path/to/jsonl).
        # Samples every SAILIR_MEMPROBE_FULL_EVERY steps (default 10) using
        # pympler.asizeof for deep-size accounting of every persistent
        # structure. See scripts/eval/archive/memprobe_full.py.
        # End-of-step malloc_trim — release glibc free top after the per-step
        # transients have been freed. Single syscall; runs every step when
        # SAILIR_END_OF_STEP_TRIM=1.
        if _end_of_step_trim is not None:
            try:
                _end_of_step_trim(0)
            except Exception:
                pass

        if _memprobe_path:
            if _memprobe_steps_set is not None:
                _should_sample = (step + 1) in _memprobe_steps_set
            else:
                _should_sample = ((step + 1) >= _memprobe_start
                                   and (step + 1) % _memprobe_every == 0)
            if _should_sample:
                try:
                    import sys as _sys
                    _archive_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'archive')
                    if _archive_dir not in _sys.path:
                        _sys.path.insert(0, _archive_dir)
                    import memprobe_full as _mpf
                    _mpf.measure(env, beam, tabu_dict, step + 1, _memprobe_path,
                                  model=model)
                except Exception as _e:
                    print(f'  [memprobe_full] error: {_e}', flush=True)

        # Focused memory breakdown. peak_rss = this step's transient peak (VmHWM
        # was reset at step start); rss = post-trim quiescent. Named structures
        # use ONE shared seen-set so they partition (no double count). The two
        # gaps separate the dominant unaccounted into its two possible causes.
        if _mem_bd and (step + 1) % _mem_bd_every == 0:
            # ── O(1) allocator-level accounting — NO object-graph walk. ────────
            # A heap walk (deep-size / gc.get_referents) needs a `seen` set of
            # one id PER LIVE OBJECT to dedup the traversal. This workload has
            # ~10^8 tiny int/set/dict objects, so that seen-set is *tens of GB*,
            # allocated every sample, larger than the heap it measures — that is
            # what ballooned rss to 146GB. So we NEVER walk objects here.
            #
            # smaps is the OS ground truth: heap+anon+file = rss EXACTLY (closes
            # by construction). malloc_info splits the glibc arena into
            # in-use/free/mmap. The CPython pymalloc arenas — where the millions
            # of small int/set/dict objects actually live — are anon bytes NOT in
            # glibc's mmap, so pymalloc_region = anon − glibc_mmap. Cost is O(1)
            # (a couple /proc reads + one malloc_info), so it cannot perturb the
            # run. It attributes rss to allocator buckets but NOT to python type/
            # source — for that, run a BOUNDED tracemalloc job (capped steps).
            _rss, _peak = _membd_mem()
            _heap, _anon, _filed = _membd_smaps()
            _rssm = _rss // 1024
            _peakm = _peak // 1024
            _heap_mb = _heap // 1024
            _anon_mb = _anon // 1024
            _file_mb = _filed // 1024
            _mi = _membd_mallinfo()
            if _mi is not None:
                _g_inuse = _mi['uordblks'] // 10**6
                _g_free = _mi['fordblks'] // 10**6
                _g_mmap = _mi['hblkhd'] // 10**6
            else:
                _g_inuse = _g_free = _g_mmap = -1
            _pyrest_mb = max(_anon_mb - max(_g_mmap, 0), 0)
            _close = _heap_mb + _anon_mb + _file_mb
            print(f'[MEMBD step {step+1}] peak_rss={_peakm}MB rss={_rssm}MB '
                  f'| smaps heap={_heap_mb}MB anon={_anon_mb}MB file={_file_mb}MB '
                  f'(sum={_close}MB={100*_close//max(_rssm,1)}% of rss) '
                  f'| glibc[inuse={_g_inuse} free={_g_free} mmap={_g_mmap}]MB '
                  f'pymalloc+torch+numpy(anon-glibcmmap)={_pyrest_mb}MB '
                  f'| n_beam={len(beam)} n_cand={len(candidates)} '
                  f'n_cache={len(env._raw_eq_cache)} '
                  f'n_bt={len(_bt_tabu) if _bt_tabu else 0}',
                  flush=True)

        # Checkpoint (rolling, overwrites) — streamed one state at a time.
        if ckpt_path and (step + 1) % ckpt_every == 0:
            _stream_dump_ckpt(ckpt_path)
            if verbose:
                print(f'  [ckpt] wrote {ckpt_path} at step {step+1}', flush=True)
        # KEPT versioned checkpoints every SAILIR_KEEP_CKPT_EVERY steps (not
        # overwritten) — lets us reconstruct the persistent state at the peak
        # step after the fact, correlated with the per-step MEMBD peak_rss log.
        _keep_every = int(os.environ.get('SAILIR_KEEP_CKPT_EVERY', '0'))
        if ckpt_path and _keep_every > 0 and (step + 1) % _keep_every == 0:
            _kp = f'{ckpt_path}.keep_step{step + 1:05d}'
            _stream_dump_ckpt(_kp)
            if verbose:
                print(f'  [keep-ckpt] wrote {_kp}', flush=True)
        # Per-step thick checkpoint (for bit-identical incremental verification)
        if ckpt_every_step and ckpt_path:
            step_path = f'{ckpt_path}.step{step + 1:04d}'
            _stream_dump_ckpt(step_path)

    return beam, best_state


# ============================================================================
# Path replay for final answer
# ============================================================================

def replay_full_expr(start_expr, path, env):
    """Replay path against the FULL start_expr (no stripping) to recover the
    complete final expression including all passenger spillover.
    """
    expr = dict(start_expr)
    subs = {}  # raw subs for replay
    for (target, ibp_op, delta) in path:
        seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        # Apply current subs to raw to get cached
        from sailir.ibp_env import apply_all_substitutions
        cached = apply_all_substitutions(raw, subs)
        if target not in cached or cached[target] == 0:
            return None
        sol = solve_ibp_for(cached, target)
        if sol is None:
            return None
        subs[target] = sol
        # Apply substitution to expr (no stripping — full)
        from sailir.ibp_env import apply_substitution as _apply
        expr = _apply(expr, target, sol)
    return expr, subs


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topology', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--integral', required=True,
                   help='Comma-separated indices for starting integral')
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--beam-width', type=int, default=40)
    p.add_argument('--top-k', type=int, default=None,
                   help='Number of model-top actions tried per (parent,target)'
                        ' per step. Default = beam_width // 2 = 20.')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--max-actions', type=int, default=900)
    p.add_argument('--beam-sort', default='mixed')
    p.add_argument('--paper-masters-only', action='store_true', default=True)
    p.add_argument('--no-paper-masters-only', dest='paper_masters_only',
                   action='store_false')
    p.add_argument('--output', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--ckpt-every', type=int, default=50)
    p.add_argument('--resume-from', default=None,
                   help='Path to a ckpt or result pkl to resume the beam from')
    p.add_argument('--tabu', action='store_true',
                   help='Enable per-expr action tabu (block re-pick of same '
                        '(target,op,delta) from same expr)')
    p.add_argument('--no-incremental-aux', dest='use_incremental_aux',
                   action='store_false', default=True,
                   help='Disable aux_flat incremental update — rebuild every '
                        'step from scratch (slower, for verification only)')
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=True,
                   help='Disable bounded-anchor exprkeyed delta — fall back to '
                        'compute_indirect_substituted_incremental (slower at '
                        'high |RS|, for verification only)')
    p.add_argument('--ckpt-every-step', action='store_true',
                   help='Write a thick per-step checkpoint after every step '
                        '(format: <ckpt_path>.stepNNNN). For bit-identical '
                        'verification of incremental-aux runs.')
    p.add_argument('--iraws-window', type=int, default=None,
                   help='If set, keep iraws anchored on the most recent N '
                        'sub_ints (recency window).')
    p.add_argument('--iraws-keep-first', type=int, default=None,
                   help='If set, keep iraws anchored on the FIRST N sub_ints '
                        '(bootstrap anchors). Empirically, drain-critical '
                        'Phase-1b hooks come from these. Combine with '
                        '--iraws-window for first-N ∪ last-M coverage.')
    p.add_argument('--no-lazy-rs', dest='lazy_rs', action='store_false',
                   default=True,
                   help='Disable LAZY_RS optimization (apply add_sub_to_resolved '
                        'eagerly for all candidates instead of only survivors). '
                        'Slower; for verification only.')
    p.add_argument('--model-batch-chunk', type=int, default=8,
                   help='If set, split the model forward batch into chunks of '
                        'this many rows. Bounds transient activation memory '
                        '(glibc retains free pages from large unchunked '
                        "forwards). None = no chunking (original behavior).")
    p.add_argument('--n-threads', type=int, default=1)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    # Memprobe init: smaps-based attribution. tracemalloc is NOT started
    # by default — tracing every Python allocation slows the whole program
    # 5-10×, even when we only sample at a few specific steps. Set
    # SAILIR_MEMPROBE_FULL_TRACEMALLOC=1 to opt in for Python-heap detail.
    _memprobe_mod = None
    if os.environ.get('SAILIR_MEMPROBE_FULL'):
        if os.environ.get('SAILIR_MEMPROBE_FULL_TRACEMALLOC') == '1':
            import tracemalloc as _tm
            _tm.start(int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_TRACEMALLOC_FRAMES', '5')))
        try:
            open(os.environ['SAILIR_MEMPROBE_FULL'], 'w').close()
        except OSError:
            pass
        import sys as _sys
        _archive_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'archive')
        if _archive_dir not in _sys.path:
            _sys.path.insert(0, _archive_dir)
        import memprobe_full as _memprobe_mod
        _memprobe_mod.record_milestone('rss_at_import_kb')

    torch.set_num_threads(args.n_threads)

    print(f'== v5 beam search ==', flush=True)
    print(f'  topology      = {args.topology}', flush=True)
    print(f'  model         = {args.model}', flush=True)
    print(f'  integral      = {args.integral}', flush=True)
    print(f'  beam_width    = {args.beam_width}', flush=True)
    print(f'  max_steps     = {args.max_steps}', flush=True)
    print(f'  max_actions   = {args.max_actions}', flush=True)
    print(f'  beam_sort     = {args.beam_sort}', flush=True)

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()
    global _V7_REGISTRY, _V7_PACKED_RS_CACHE
    _V7_REGISTRY = IntegralRegistry()
    _V7_PACKED_RS_CACHE = {}
    if _memprobe_mod is not None:
        _memprobe_mod.record_milestone('rss_after_env_kb')

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    if _memprobe_mod is not None:
        _memprobe_mod.record_milestone('rss_after_model_kb')
        _memprobe_mod.record_model(model)

    # Strip surrounding quotes that may survive condor argv parsing
    integral_str = args.integral.strip("'").strip('"')
    start_int = tuple(int(x) for x in integral_str.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    print(f'  start integral weight = {start_w}', flush=True)
    print(f'  active threshold      = (w1,w2) >= {start_w12}', flush=True)

    target_sector = tuple(get_sector_mask(start_int))
    print(f'  target_sector = {target_sector}', flush=True)

    start_expr = {start_int: 1}

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        beam_sort=args.beam_sort,
        max_actions=args.max_actions,
        ckpt_path=args.ckpt,
        ckpt_every=args.ckpt_every,
        resume_from=args.resume_from,
        tabu=args.tabu,
        use_incremental_aux=args.use_incremental_aux,
        use_exprkeyed=args.use_exprkeyed,
        ckpt_every_step=args.ckpt_every_step,
        iraws_window=args.iraws_window,
        iraws_keep_first=args.iraws_keep_first,
        lazy_rs=args.lazy_rs,
        model_batch_chunk=args.model_batch_chunk,
        top_k=args.top_k,
    )

    print(f'\n== DONE — beam size {len(beam)} ==', flush=True)
    print(f'best state: n_non_masters={best_state.n_non_masters} '
          f'mw={max_w12(best_state.expr, target_sector)} '
          f'path_len={len(best_state.path)} '
          f'score={best_state.score:.4f}', flush=True)
    # Peak RSS (kernel-tracked, monotonic over the process lifetime).
    try:
        with open('/proc/self/status') as _ps:
            for _line in _ps:
                if _line.startswith('VmHWM:'):
                    _peak_kb = int(_line.split()[1])
                    print(f'peak_rss_kb={_peak_kb} ({_peak_kb // 1024} MB)',
                          flush=True)
                    break
    except OSError:
        pass

    if best_state.n_non_masters == 0:
        print('SUCCESS — active bucket drained (start_int = passenger expansion)', flush=True)
    else:
        print('INCOMPLETE — active bucket still has non-masters', flush=True)

    if args.output:
        # Replay full expr (with passenger) for the best state
        print('Replaying path for full final expression...', flush=True)
        full_result = replay_full_expr(start_expr, best_state.path, env)
        out = {
            'best_state': best_state._asdict(),
            'beam': [s._asdict() for s in beam],
            'start_integral': start_int,
            'start_w12': start_w12,
            'target_sector': target_sector,
            'full_expr_replay': full_result[0] if full_result else None,
            'full_subs_replay': full_result[1] if full_result else None,
        }
        with open(args.output, 'wb') as f:
            pickle.dump(out, f)
        print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
