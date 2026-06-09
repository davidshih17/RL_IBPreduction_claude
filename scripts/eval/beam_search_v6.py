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


def _aux_to_result(aux_flat):
    """Reconstruct compute_indirect_substituted result list from aux tuple."""
    cached_unique, union_bms, raw_id_to_idx, indirect_raws = aux_flat
    if not indirect_raws:
        return []
    result = []
    for sub_int, ibp_op, shift, raw in indirect_raws:
        idx = raw_id_to_idx[id(raw)]
        result.append((sub_int, ibp_op, shift, raw,
                       cached_unique[idx], union_bms[idx]))
    return result


_PICKLABLE_AUX_MARKER = '__v5_aux_v2__'


def _aux_to_picklable(aux_flat):
    """Convert aux's rid from id(raw)-keyed (process-local, doesn't survive
    pickle) to (ibp_op, seed)-keyed (stable across processes).

    Each raw is uniquely determined by the (ibp_op, seed) that produced it:
    seed = sub_int - shift. So we can use that as a stable key.

    Returns a tuple where the last element is a marker so loaders can
    distinguish old/new format.
    """
    if aux_flat is None:
        return None
    cu, ubm, rid, iraws = aux_flat
    # Build id() → (op, seed) mapping by walking iraws
    id_to_opseed = {}
    for sub_int, op, shift, raw in iraws:
        if id(raw) in id_to_opseed:
            continue
        n = len(sub_int)
        seed = tuple(sub_int[i] - shift[i] for i in range(n))
        id_to_opseed[id(raw)] = (op, seed)
    # Build stable rid
    stable_rid = {id_to_opseed[old_id]: idx for old_id, idx in rid.items()}
    # Note: iraws still has Python id()-via-pickle issues for raw objects,
    # but that doesn't matter — on load we'll re-key rid by walking iraws
    # in current process. cu/ubm content is preserved as-is.
    return (cu, ubm, stable_rid, iraws, _PICKLABLE_AUX_MARKER)


def _aux_from_picklable(aux_flat, env=None):
    """Convert aux from picklable (op, seed)-keyed form back to id(raw)-keyed.
    Returns the standard 4-tuple aux usable by all helpers.

    If env is provided, replace each iraws entry's raw with env's cached
    version (via env.get_raw_equation_cached). Critical for resume bit-
    identical behavior: env._raw_eq_cache is process-local, so pickle-
    loaded raws are DIFFERENT objects than the ones env will hand out for
    the same (op, seed). Future incremental updates fetch raws from env,
    leading to inconsistent id() mappings. Unifying via env eliminates this.
    """
    if aux_flat is None:
        return None
    # Detect new format by length + trailing marker
    if len(aux_flat) == 5 and aux_flat[4] == _PICKLABLE_AUX_MARKER:
        cu, ubm, stable_rid, iraws, _ = aux_flat
        new_iraws = iraws
        if env is not None:
            # Replace each iraws entry's raw with env's cached version.
            new_iraws = []
            for sub_int, op, shift, raw in iraws:
                n = len(sub_int)
                seed = tuple(sub_int[i] - shift[i] for i in range(n))
                env_raw = env.get_raw_equation_cached(op, seed)
                new_iraws.append((sub_int, op, shift, env_raw))
        # Build id()-keyed rid using the (possibly env-unified) raws
        new_rid = {}
        for sub_int, op, shift, raw in new_iraws:
            k = id(raw)
            if k in new_rid:
                continue
            n = len(sub_int)
            seed = tuple(sub_int[i] - shift[i] for i in range(n))
            cu_idx = stable_rid.get((op, seed))
            if cu_idx is None:
                raise ValueError(
                    f'aux load: no stable_rid entry for (op={op}, seed={seed})'
                )
            new_rid[k] = cu_idx
        return (cu, ubm, new_rid, new_iraws)
    # Legacy 4-tuple format — assume same-process; just pass through.
    return aux_flat


def _prune_aux_by_recency(aux_flat, recent_sub_ints):
    """Drop iraws entries whose sub_int (anchor) isn't in recent_sub_ints.
    Rebuild cu/ubm/rid from the kept raws.

    Args:
        aux_flat: (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
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
    # Rebuild cu/ubm/rid: only keep entries referenced by surviving iraws.
    referenced = set()
    for e in new_iraws:
        referenced.add(id(e[3]))
    new_cu = []
    new_ubm = []
    new_rid = {}
    for old_id, old_idx in rid.items():
        if old_id in referenced:
            new_rid[old_id] = len(new_cu)
            new_cu.append(cu[old_idx])
            new_ubm.append(ubm[old_idx])
    return (new_cu, new_ubm, new_rid, new_iraws)
# DO NOT `from ... import PRIME` — set_prime() rebinds the module attr but
# not local bindings. Always read ibp_env.PRIME so we pick up the configured value.
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_non_masters, get_sector_mask, filter_to_sector


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
    if use_exprkeyed:
        from sailir.ibp_env import compute_indirect_substituted_exprkeyed_delta
        # Pass child's expr_nm (target-sector non-masters) so the bounded
        # anchor set is computed correctly.
        expr_nm = list(get_non_masters(child.expr, target_sector).keys())
        _result, new_aux = compute_indirect_substituted_exprkeyed_delta(
            parent.aux_flat, expr_nm, target, new_resolved_sol,
            child.resolved_subs,
            env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
    else:
        from sailir.ibp_env import compute_indirect_substituted_incremental
        _result, new_aux = compute_indirect_substituted_incremental(
            parent.aux_flat, target, new_resolved_sol,
            child.resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
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
                   lazy_rs=True):
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

    def sort_weight(s):
        return (s.max_w12, s.n_non_masters, -s.score)

    def sort_totalweight(s):
        return (s.total_w12, s.n_non_masters, -s.score)

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
                indirect_cache = _aux_to_result(s.aux_flat)
                if _v5_prof:
                    _ct['aux_repack_calls'] += 1
            else:
                if use_exprkeyed:
                    from sailir.ibp_env import compute_indirect_substituted_exprkeyed
                    indirect_cache, new_aux = compute_indirect_substituted_exprkeyed(
                        list(nm.keys()), s.resolved_subs,
                        env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
                    )
                else:
                    # Optionally restrict the fresh rebuild's sub-int keyset.
                    if (iraws_window is not None or iraws_keep_first is not None):
                        keys = list(s.resolved_subs.keys())
                        keep = set()
                        if iraws_keep_first is not None:
                            keep.update(keys[:iraws_keep_first])
                        if iraws_window is not None:
                            keep.update(keys[-iraws_window:])
                        if len(keep) < len(keys):
                            keep_list = [k for k in keys if k in keep]
                            fresh_rs = {k: s.resolved_subs[k] for k in keep_list}
                            fresh_dummy = {k: {} for k in keep_list}
                        else:
                            fresh_rs = s.resolved_subs
                            fresh_dummy = dummy_subs
                    else:
                        fresh_rs = s.resolved_subs
                        fresh_dummy = dummy_subs
                    indirect_cache, new_aux = compute_indirect_substituted_with_aux(
                        fresh_dummy, fresh_rs, env.ibp_t, env.li_t, env.shifts,
                        env._raw_eq_cache,
                    )
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
                valid = enumerate_valid_actions_with_indirect_cache(
                    target, indirect_cache, dummy_subs, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts, 'subsector',
                    env._raw_eq_cache,
                )
                if _v5_prof:
                    _p1['gv'] += time.time() - _t
                    _ct['enum_calls'] += 1
                # Tabu: filter out (target, op, delta) tried previously from
                # this expr fingerprint.
                if tabu_dict is not None and valid:
                    _t = time.time() if _v5_prof else 0
                    expr_fp = frozenset(s.expr.items())
                    tabu_set = tabu_dict.get(expr_fp)
                    if tabu_set:
                        valid = [(op, dlt) for (op, dlt) in valid
                                 if (target, op, dlt) not in tabu_set]
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
            if verbose:
                print(f'[v6 step {step}] no tasks — STUCK', flush=True)
            break

        # Run model batched on all tasks
        _t = time.time() if _v5_prof else 0
        batch_data = []
        for parent_idx, target, valid in tasks:
            s = beam[parent_idx]
            batch_data.append((s.expr, s.resolved_subs, valid, target_sector, target))
        b = prepare_batched_input_v5_dummy(batch_data, device,
                                           max_actions=max_actions)
        if _v5_prof:
            _p2['batch_prep'] += time.time() - _t
            _t = time.time()
        with torch.no_grad():
            logits, probs = model(
                b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                b['sub_repl_mask'], b['sub_mask'],
                b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                b['sector_mask'], b['target_integral'],
            )
        if _v5_prof:
            _p2['model_fwd'] += time.time() - _t

        # For each task, take top-K actions by prob, apply, generate candidates
        K = max(1, beam_width // 2)
        for ti, (parent_idx, target, valid) in enumerate(tasks):
            n_v = min(len(valid), probs.shape[1])
            row = probs[ti, :n_v].numpy()
            # Take top-K by prob
            top_idx = np.argsort(-row)[:K]
            parent_state = beam[parent_idx]
            # Tabu: record (target, op, delta) we choose to try from this expr
            if tabu_dict is not None:
                expr_fp = frozenset(parent_state.expr.items())
                ts = tabu_dict.setdefault(expr_fp, set())
                for ai in top_idx:
                    op, delta = valid[int(ai)]
                    ts.add((target, op, delta))
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

        def _serialize_beam_for_ckpt():
            out = []
            for s in beam:
                sd = s._asdict()
                if sd.get('aux_flat') is not None:
                    sd['aux_flat'] = _aux_to_picklable(sd['aux_flat'])
                out.append(sd)
            return out

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

        # Checkpoint (rolling, overwrites)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            with open(ckpt_path, 'wb') as f:
                pickle.dump({
                    'step': step + 1,
                    'beam': _serialize_beam_for_ckpt(),
                    'target_sector': target_sector,
                    'start_w12': start_w12,
                    'tabu_dict': _serialize_tabu_for_ckpt(),
                }, f)
            if verbose:
                print(f'  [ckpt] wrote {ckpt_path} at step {step+1}', flush=True)
        # Per-step thick checkpoint (for bit-identical incremental verification)
        if ckpt_every_step and ckpt_path:
            step_path = f'{ckpt_path}.step{step + 1:04d}'
            with open(step_path, 'wb') as f:
                pickle.dump({
                    'step': step + 1,
                    'beam': _serialize_beam_for_ckpt(),
                    'target_sector': target_sector,
                    'start_w12': start_w12,
                    'tabu_dict': _serialize_tabu_for_ckpt(),
                }, f)

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
    p.add_argument('--n-threads', type=int, default=1)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

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

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

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
    )

    print(f'\n== DONE — beam size {len(beam)} ==', flush=True)
    print(f'best state: n_non_masters={best_state.n_non_masters} '
          f'mw={max_w12(best_state.expr, target_sector)} '
          f'path_len={len(best_state.path)} '
          f'score={best_state.score:.4f}', flush=True)

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
