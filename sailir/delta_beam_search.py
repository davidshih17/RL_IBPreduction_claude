"""Serial (n_workers=1), delta-tracking beam search.

Every beam state stores ONLY (parent_ref, delta=(target, sol_target)). All
heavy fields are recomputed incrementally from parent + delta — no full
rebuilds from scratch each step.

Design (per user pref: optimize for SPEED, memory secondary):
  - Eager at candidate construction: expr, max_w, n_non_masters, score.
    These are cheap and used by sort/eval.
  - Lazy via @property:
        subs, resolved_subs   — needed by P1 of the NEXT step (survivors only)
                                 and by DEDUP_VARIANT=rs key (in sort order, so
                                 only the top ~beam_width candidates get hit)
        indirect_aux, indirect_cache_list — built by
                                 compute_indirect_substituted_incremental from
                                 the parent's aux; only paid for survivors.
  - End-of-step memory cleanup: for each survivor in the new beam, drop the
    parent's heavy fields (expr/subs/resolved_subs/aux). Path reconstruction
    needs only parent.action + parent.parent, which we never clear.

  This means *no IPC, no pickle dance* (worker pool overhead is what killed
  the original incremental-aux idea), and live memory is bounded to roughly
  one beam's worth of states.
"""
import math
import random
import time

import torch

from sailir import ibp_env  # for live N_INDICES / PRIME after init_from_topology
from sailir.ibp_env import (
    is_master, weight,
    apply_resolved_subs, solve_ibp_for, apply_substitution_target_only,
    add_sub_to_resolved,
    compute_indirect_substituted_incremental,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
    integral_in_exact_sector,
)

from beam_search_utils import get_non_masters


# =============================================================================
# DeltaState
# =============================================================================
class DeltaState:
    """Beam state defined by (parent, delta) plus eager-cheap fields."""
    __slots__ = (
        'parent', 'action', 'score', 'path_len',
        'expr', 'max_w', 'n_non_masters',
        '_delta_target', '_delta_sol',
        '_subs', '_resolved_subs',
        '_indirect_aux', '_indirect_cache_list',
        '_env', '_target_sector',
        '_expr_key',
        '_resumed_path',
    )

    def __init__(self, parent, delta_target, delta_sol, action, score, env, target_sector,
                 expr, max_w, n_non_masters,
                 subs=None, resolved_subs=None, indirect_aux=None,
                 indirect_cache_list=None):
        self.parent = parent
        self._delta_target = delta_target
        self._delta_sol = delta_sol
        self.action = action
        self.score = score
        self.path_len = (parent.path_len + 1) if parent is not None else 0
        self.expr = expr
        self.max_w = max_w
        self.n_non_masters = n_non_masters
        self._subs = subs
        self._resolved_subs = resolved_subs
        self._indirect_aux = indirect_aux
        self._indirect_cache_list = indirect_cache_list
        self._env = env
        self._target_sector = target_sector
        self._expr_key = None
        # Set after checkpoint-resume so reconstruct_path() returns the full
        # action chain even though parent=None at the resumed roots.
        self._resumed_path = None

    # ------------------------------------------------------------------------
    # Root state
    # ------------------------------------------------------------------------
    @classmethod
    def root(cls, env, start_expr, target_sector):
        expr = dict(start_expr)
        mw, nm = _compute_max_w_and_nm(expr)
        return cls(
            parent=None, delta_target=None, delta_sol=None, action=None,
            score=0.0, env=env, target_sector=target_sector,
            expr=expr, max_w=mw, n_non_masters=nm,
            subs={}, resolved_subs={},
            indirect_aux=([], [], {}, []),
            indirect_cache_list=[],
        )

    # ------------------------------------------------------------------------
    # Resume-stub root (rebuilt from a checkpoint)
    # ------------------------------------------------------------------------
    @classmethod
    def from_checkpoint_dict(cls, env, target_sector, d):
        """Rebuild a self-contained DeltaState from its checkpoint dict.

        `d` carries: expr, max_w, n_non_masters, score, path_len, path,
        subs, resolved_subs, indirect_aux (cu, ubm, rid, iraws),
        indirect_cache_list. The resulting state has parent=None,
        action=None, so the next P3 step builds children from it as
        though it were a fresh root.
        """
        s = cls(
            parent=None, delta_target=None, delta_sol=None, action=None,
            score=float(d['score']), env=env, target_sector=target_sector,
            expr=dict(d['expr']), max_w=tuple(d['max_w']),
            n_non_masters=int(d['n_non_masters']),
            subs=dict(d['subs']),
            resolved_subs=dict(d['resolved_subs']),
            indirect_aux=d['indirect_aux'],
            indirect_cache_list=d['indirect_cache_list'],
        )
        s.path_len = int(d['path_len'])
        s._resumed_path = tuple(d['path']) if d.get('path') else ()
        return s

    # ------------------------------------------------------------------------
    # Lazy fields (materialized on first access; folded into self's slots)
    # ------------------------------------------------------------------------
    @property
    def subs(self):
        if self._subs is None:
            new = dict(self.parent.subs)
            new[self._delta_target] = self._delta_sol
            self._subs = new
        return self._subs

    @property
    def resolved_subs(self):
        if self._resolved_subs is None:
            self._resolved_subs = add_sub_to_resolved(
                self.parent.resolved_subs, self._delta_target, self._delta_sol,
            )
        return self._resolved_subs

    @property
    def indirect_aux(self):
        if self._indirect_aux is None:
            self._materialize_aux()
        return self._indirect_aux

    @property
    def indirect_cache_list(self):
        if self._indirect_cache_list is None:
            self._materialize_aux()
        return self._indirect_cache_list

    def _materialize_aux(self):
        env = self._env
        import os as _os_sec
        _sec_proj = _os_sec.environ.get('DELTA_SECTOR_PROJECT_AUX', '0') == '1'
        _ts = self._target_sector if _sec_proj else None
        # Provably-lossless bounded-memory aux: rebuild from CURRENT expr_nm
        # + USEFUL resolved_sub anchors (those with sol_K ∩ expr_nm ≠ ∅).
        # Opt-in via DELTA_IRAWS_EXPRKEYED=1. See docstring of
        # compute_indirect_substituted_exprkeyed for the (A)+(B) coverage
        # proof using the fact that resolved_subs is fully flattened.
        _exprkeyed = _os_sec.environ.get('DELTA_IRAWS_EXPRKEYED', '0') == '1'
        _debug_compare = _os_sec.environ.get('DELTA_DEBUG_COMPARE', '0') == '1'
        _debug_start = int(_os_sec.environ.get('DELTA_DEBUG_COMPARE_START_STEP', '0'))
        if _debug_compare and self.path_len < _debug_start:
            _debug_compare = False
        if _exprkeyed:
            # Bounded-anchor INCREMENTAL delta: Phase 0 (drop iraws of
            # leaving anchors) + Phase A (substitute new_sub_int) + Phase B
            # (add iraws for entering anchors). cu/iraws bounded by
            # |anchors| = |expr_nm| + |useful K| (not by depth).
            parent_rs = self.parent.resolved_subs
            new_resolved_sol = apply_resolved_subs(self._delta_sol, parent_rs)
            new_rs = self.resolved_subs  # force materialization
            from beam_search_utils import get_non_masters as _get_nm
            expr_nm = _get_nm(self.expr, self._target_sector)
            from sailir.ibp_env import compute_indirect_substituted_exprkeyed_delta as _eb
            result, aux = _eb(
                self.parent.indirect_aux, expr_nm.keys(),
                self._delta_target, new_resolved_sol, new_rs,
                env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
                target_sector=_ts,
            )
            if _debug_compare:
                # Parallel from-scratch baseline rebuild: enumerate ALL past
                # sub_ints (not just useful K's) and compare the resulting
                # iraws/cached against v4's.
                # IMPORTANT: pass _ts (the SAME target_sector v4 received,
                # = None when DELTA_SECTOR_PROJECT_AUX=0). Mismatched sector
                # args make baseline apply projection that v4 didn't, giving
                # spurious "v4 has extras" reports. expr_nm in the compare
                # ALWAYS uses self._target_sector (real sector for filtering
                # the action-target set, distinct from cu projection).
                _diff_v4_vs_baseline(
                    self, result, aux, new_rs, env, _ts,
                    expr_target_sector=self._target_sector,
                )
            self._indirect_aux = aux
            self._indirect_cache_list = result
            return
        # Original delta-tracking incremental update from parent.aux.
        parent_rs = self.parent.resolved_subs
        new_resolved_sol = apply_resolved_subs(self._delta_sol, parent_rs)
        new_rs = self.resolved_subs  # force materialization
        result, aux = compute_indirect_substituted_incremental(
            self.parent.indirect_aux, self._delta_target, new_resolved_sol, new_rs,
            env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
            target_sector=_ts,
        )
        self._indirect_aux = aux
        self._indirect_cache_list = result

    # ------------------------------------------------------------------------
    # expr fingerprint for action-level tabu (lazy, cached)
    # ------------------------------------------------------------------------
    @property
    def expr_key(self):
        """Frozen fingerprint of the current expression for tabu indexing.
        Cached lazily; computed only when DELTA_TABU=1 actually touches it.
        Must be read while self.expr is still alive (i.e., before end-of-step
        P4 cleanup nulls it on dropped parents). Zero-coefficient entries are
        dropped so two paths reaching the same effective expression collide.
        """
        if self._expr_key is None:
            self._expr_key = frozenset(
                (k, v) for k, v in self.expr.items() if v != 0
            )
        return self._expr_key

    # ------------------------------------------------------------------------
    # Path reconstruction (walks chain — only at finalization)
    # ------------------------------------------------------------------------
    def reconstruct_path(self):
        actions = []
        s = self
        while s.parent is not None:
            actions.append(s.action)
            s = s.parent
        actions.reverse()
        # If we hit a resume-stub root, prepend its frozen action prefix so
        # the caller sees the full path including pre-resume history.
        if s._resumed_path:
            return list(s._resumed_path) + actions
        return actions

    @property
    def path(self):
        """API-compatibility: production State.path is a list. Lazy build."""
        return self.reconstruct_path()


# =============================================================================
# Helpers
# =============================================================================

# One-shot guard: once we print the first divergence, subsequent _materialize_aux
# calls within the same process should NOT keep dumping (otherwise we drown the
# log). State + step + first action mismatch is sufficient.
_DEBUG_COMPARE_FIRED = False


def _diff_v4_vs_baseline(state, v4_result, v4_aux, new_rs, env, target_sector,
                          expr_target_sector=None):
    """Parallel from-scratch baseline rebuild + element-wise compare.

    Compares the v4 exprkeyed-delta result against the depth-keyed
    `compute_indirect_substituted_with_aux` baseline (anchor on ALL past
    sub_ints). For each step:
      1. Build baseline iraws via the from-scratch full compute.
      2. Build (op, seed) → cached  dicts from both v4 and baseline.
      3. Print first key set difference + first cached content difference.

    This pinpoints the missed-action bug exactly: if baseline has (op, seed)
    that v4 lacks, the anchor for that raw is missing from v4's anchor set;
    the print will show the (anchor, op, shift, raw_keys, cached_keys) tuple
    so we can identify which K was not anchored.

    Per-step cost: O(|RS|*|shifts|) for the baseline rebuild. Intended ONLY
    for instrumented probe runs with DELTA_DEBUG_COMPARE=1.
    """
    global _DEBUG_COMPARE_FIRED
    if _DEBUG_COMPARE_FIRED:
        return
    from sailir.ibp_env import compute_indirect_substituted_with_aux

    # Build subs list (past sub_ints) from new_rs keys — same input as the
    # depth-keyed baseline gets from production beam_search_full.
    subs = list(new_rs.keys())
    base_result, base_aux = compute_indirect_substituted_with_aux(
        subs, new_rs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        target_sector=target_sector,
    )

    # Index both result lists by (op, seed). Multiple entries can share an
    # (op, seed) when different anchors generate the same raw; the cached is
    # the same so we keep the first occurrence.
    def _index(result_list):
        idx = {}
        for entry in result_list:
            sub_int, ibp_op, shift, raw, cached, union_bm = entry
            seed = tuple(sub_int[i] - shift[i] for i in range(ibp_env.N_INDICES))
            key = (ibp_op, seed)
            if key not in idx:
                idx[key] = (sub_int, raw, cached, union_bm)
        return idx

    v4_idx = _index(v4_result)
    base_idx = _index(base_result)

    v4_keys = set(v4_idx.keys())
    base_keys = set(base_idx.keys())
    missed_by_v4 = base_keys - v4_keys
    extra_in_v4 = v4_keys - base_keys
    common_keys = v4_keys & base_keys

    # CACHED EQUALITY CHECK for common (op, seed) keys. If v4's incremental
    # Phase A drifts (missed a substitution), the cached dict differs from
    # baseline's from-scratch cached. This is a SILENT bug that key-set
    # comparison cannot detect.
    cached_mismatches = []
    for key in common_keys:
        v4_cached = v4_idx[key][2]
        base_cached = base_idx[key][2]
        # Compare as dicts (order-insensitive)
        if v4_cached != base_cached:
            cached_mismatches.append(key)
            if len(cached_mismatches) >= 5:
                break

    # We only care about iraws entries that actually emit a Phase 1b action
    # against SOME current target. For diagnosis, we report ALL key-set
    # differences, then mark which ones actually pass SC1+SC2 for any of
    # the current expr non-masters.
    # expr_target_sector is the REAL target sector (used for filtering the
    # action-target set in production). target_sector is the OPTIONAL cu
    # projection (None when DELTA_SECTOR_PROJECT_AUX=0).
    _expr_ts = expr_target_sector if expr_target_sector is not None else target_sector
    from beam_search_utils import get_non_masters as _get_nm
    expr_nm = _get_nm(state.expr, _expr_ts)
    targets = list(expr_nm.keys())

    def _emits_action(raw, cached):
        for T in targets:
            in_raw = (T in raw and raw[T] != 0)
            in_cached = (T in cached and cached[T] != 0)
            if (not in_raw) and in_cached:
                return T
        return None

    step = state.path_len
    # Diagnostic: also count |state.expr| (raw size including masters) and
    # |state.expr non-zero| to understand expr_nm=0 cases.
    expr_size = len(state.expr) if state.expr is not None else -1
    expr_nz = sum(1 for v in state.expr.values() if v != 0) if state.expr else 0
    print(f"[DBG_COMPARE step={step}] v4_iraws={len(v4_result)} "
          f"base_iraws={len(base_result)} "
          f"v4_keys={len(v4_keys)} base_keys={len(base_keys)} "
          f"missed_by_v4={len(missed_by_v4)} extra_in_v4={len(extra_in_v4)} "
          f"cached_mismatches={len(cached_mismatches)} "
          f"|RS|={len(new_rs)} |expr|={expr_size}({expr_nz}nz) "
          f"|expr_nm|={len(targets)}", flush=True)

    # Fire on first cached mismatch \u2014 this is a Phase A staleness bug.
    if cached_mismatches:
        # Check whether any of the mismatched cached entries actually emit
        # action against current targets (more diagnostic than just count).
        action_mismatches = []
        for key in cached_mismatches:
            v4_sub, v4_raw, v4_cached, _ = v4_idx[key]
            base_sub, base_raw, base_cached, _ = base_idx[key]
            for T in targets:
                in_raw = (T in v4_raw and v4_raw[T] != 0)
                v4_in_cached = (T in v4_cached and v4_cached[T] != 0)
                base_in_cached = (T in base_cached and base_cached[T] != 0)
                if (not in_raw) and (v4_in_cached != base_in_cached):
                    action_mismatches.append((key, T, v4_in_cached, base_in_cached))
                    break
        if action_mismatches:
            print(f"[DBG_COMPARE step={step}] *** CACHED ACTION MISMATCH: "
                  f"{len(action_mismatches)} keys differ action emission ***",
                  flush=True)
            for i, (key, T, v4_emit, base_emit) in enumerate(action_mismatches[:3]):
                op, seed = key
                v4_sub, v4_raw, v4_cached, _ = v4_idx[key]
                _, base_raw, base_cached, _ = base_idx[key]
                # Sanity check: same raw between v4 and baseline?
                same_raw = (v4_raw == base_raw)
                # Diff cached keys
                v4_set = set(v4_cached.keys())
                base_set = set(base_cached.keys())
                in_v4_only = sorted(v4_set - base_set)
                in_base_only = sorted(base_set - v4_set)
                # Categorize in_v4_only by RS membership
                in_v4_only_in_rs = [k for k in in_v4_only if k in new_rs]
                in_v4_only_not_in_rs = [k for k in in_v4_only if k not in new_rs]
                # Also: are these terms in the RAW originally?
                in_v4_only_in_raw = [k for k in in_v4_only if k in v4_raw]
                # And: what does apply_resolved_subs do on the raw fresh?
                from sailir.ibp_env import apply_resolved_subs as _ars
                from sailir.ibp_env import apply_resolved_subs_batch as _arsb
                fresh = _ars(v4_raw, new_rs)
                fresh_batch = _arsb([v4_raw], new_rs)[0]
                fresh_matches_base = (fresh == base_cached)
                fresh_batch_matches_base = (fresh_batch == base_cached)
                fresh_vs_fresh_batch = (fresh == fresh_batch)
                print(
                    f"  [#{i}] op={op} seed={seed}\n"
                    f"        T={T}  v4_emit={v4_emit} base_emit={base_emit}\n"
                    f"        same_raw_v4_vs_base={same_raw}  |raw|={len(v4_raw)}\n"
                    f"        |v4_cached|={len(v4_cached)} |base_cached|={len(base_cached)}\n"
                    f"        fresh_ars_matches_base={fresh_matches_base} |fresh_ars|={len(fresh)}\n"
                    f"        fresh_arsbatch_matches_base={fresh_batch_matches_base} "
                    f"|fresh_arsbatch|={len(fresh_batch)}\n"
                    f"        fresh_ars_vs_arsbatch_match={fresh_vs_fresh_batch}\n"
                    f"        |in_v4_only|={len(in_v4_only)} (in_RS={len(in_v4_only_in_rs)}, "
                    f"not_in_RS={len(in_v4_only_not_in_rs)}, "
                    f"in_raw_orig={len(in_v4_only_in_raw)})\n"
                    f"        in_v4_only_in_RS_first3={in_v4_only_in_rs[:3]}\n"
                    f"        in_v4_only_not_in_RS_first3={in_v4_only_not_in_rs[:3]}",
                    flush=True,
                )
            _DEBUG_COMPARE_FIRED = True
            return
        # Cached mismatches exist but don't change action emission for
        # current targets. Still note this is a silent staleness bug.
        if step % 10 == 0:
            print(f"[DBG_COMPARE step={step}] cached_mismatches={len(cached_mismatches)} "
                  f"but none affect current target action emission",
                  flush=True)

    # Find action-emitting missing entries first
    action_emitting_missing = []
    for key in missed_by_v4:
        sub_int, raw, cached, ubm = base_idx[key]
        T_hit = _emits_action(raw, cached)
        if T_hit is not None:
            action_emitting_missing.append((key, sub_int, raw, cached, T_hit))

    if action_emitting_missing:
        print(f"[DBG_COMPARE step={step}] *** ACTION-EMITTING MISSED ENTRIES: "
              f"{len(action_emitting_missing)} ***", flush=True)
        # Dump the first 3
        for i, (key, sub_int, raw, cached, T_hit) in enumerate(action_emitting_missing[:3]):
            ibp_op, seed = key
            # Which K in raw is responsible for bringing T_hit into cached?
            culprit_K = None
            for K in raw:
                if raw[K] == 0:
                    continue
                if K in new_rs:
                    sol_K = new_rs[K]
                    if T_hit in sol_K and sol_K[T_hit] != 0:
                        culprit_K = K
                        break
            sol_culprit_in_expr = None
            if culprit_K is not None:
                sol_culprit = new_rs[culprit_K]
                sol_culprit_in_expr = sum(1 for k in sol_culprit if k in expr_nm)
            # Was the anchor sub_int in v4's anchor set?
            sub_int_in_v4_anchors = False
            if sub_int in new_rs:
                sol_anchor = new_rs[sub_int]
                sub_int_in_v4_anchors = any(k in expr_nm for k in sol_anchor)
            print(
                f"  [#{i}] op={ibp_op} seed={seed}\n"
                f"        anchor_sub_int={sub_int} (in_v4_anchors={sub_int_in_v4_anchors})\n"
                f"        target_introduced={T_hit}\n"
                f"        raw |keys|={len(raw)} raw_K_in_RS_count="
                f"{sum(1 for K in raw if K in new_rs and raw[K] != 0)}\n"
                f"        culprit_K (K in raw with T_hit in sol_K)={culprit_K}\n"
                f"        sol_culprit ∩ expr_nm count={sol_culprit_in_expr}",
                flush=True,
            )
        _DEBUG_COMPARE_FIRED = True
        return

    if missed_by_v4 and (step % 25 == 0):
        # Missing keys exist but none emit a Phase 1b action against current
        # targets — these are "free" entries that don't actually matter for
        # this step's action set. Log a brief sample every 25 steps.
        print(f"[DBG_COMPARE step={step}] missed_by_v4={len(missed_by_v4)} "
              f"but none emit actions on current expr_nm (harmless this step)",
              flush=True)
    if extra_in_v4:
        # v4 should never have extras — would be a serious bug.
        sample = list(extra_in_v4)[:3]
        print(f"[DBG_COMPARE step={step}] *** EXTRA IN V4: {len(extra_in_v4)} "
              f"sample={sample} ***", flush=True)
        _DEBUG_COMPARE_FIRED = True


def _compute_max_w_and_nm(expr):
    nm = 0
    mw = (0, 0)
    for k, v in expr.items():
        if v == 0:
            continue
        if not is_master(k):
            nm += 1
            w = weight(k)
            wp = (w[0], w[1])
            if wp > mw:
                mw = wp
    return mw, nm


def _sort_key(s):
    """Production '--beam-sort weight' sort key."""
    return (s.max_w, s.n_non_masters, -s.score)


def _dedup_key_rs(s):
    """Production DEDUP_VARIANT=rs fingerprint."""
    return frozenset(
        (sk, frozenset(sv.items()))
        for sk, sv in s.resolved_subs.items()
    )


# =============================================================================
# Apply one action → child DeltaState (eager-cheap fields, lazy-heavy)
# =============================================================================
def _apply_action(state, target, ibp_op, delta_shift, action_prob, target_sector):
    env = state._env
    seed = tuple(target[i] + delta_shift[i] for i in range(ibp_env.N_INDICES))
    raw = env.get_raw_equation_cached(ibp_op, seed)
    cached = apply_resolved_subs(raw, state.resolved_subs)
    if target not in cached or cached[target] == 0:
        return None
    sol = solve_ibp_for(cached, target)
    if sol is None:
        return None
    sol_target = {k: v for k, v in sol.items()
                  if integral_in_exact_sector(k, target_sector)}
    new_expr = apply_substitution_target_only(state.expr, target, sol_target, target_sector)
    mw, nm = _compute_max_w_and_nm(new_expr)
    new_score = state.score + math.log(action_prob + 1e-10)
    return DeltaState(
        parent=state,
        delta_target=target,
        delta_sol=sol_target,
        action=(target, ibp_op, delta_shift),
        score=new_score,
        env=env,
        target_sector=target_sector,
        expr=new_expr,
        max_w=mw,
        n_non_masters=nm,
    )


def _replay_path(env, target_sector, start_expr, path, saved_score):
    """Replay a list of (target, ibp_op, delta_shift) actions starting
    from `start_expr` to deterministically rebuild a full DeltaState.

    Used by thin-checkpoint resume: instead of serializing the full aux
    on save, we serialize only the action paths, then re-execute them
    here. This skips P1+P2 (no model calls; we already know which actions
    to take), so replay is ~10x faster than the original beam search.

    The replayed state is then flattened to a resumed-root (parent=None,
    _resumed_path set) so reconstruct_path() returns the full pre-resume
    action chain and the parent chain can be garbage-collected.
    """
    state = DeltaState.root(env, start_expr, target_sector)
    for (target, ibp_op, delta_shift) in path:
        new_state = _apply_action(
            state, target, ibp_op, delta_shift,
            action_prob=1.0, target_sector=target_sector,
        )
        if new_state is None:
            raise RuntimeError(
                f'Replay failed at action {(target, ibp_op, delta_shift)} '
                f'(step {state.path_len})'
            )
        # Force materialization so the next action's _apply_action sees
        # the parent's resolved_subs and aux.
        _ = new_state.subs
        _ = new_state.resolved_subs
        _ = new_state.indirect_cache_list
        state = new_state
    # Flatten: drop the chain and mark as a resumed-root.
    state.parent = None
    state.action = None
    state.score = float(saved_score)
    state._resumed_path = tuple(path)
    return state


def _apply_actions_parallel_fork(work, tasks, target_sector, n_workers):
    """Apply a list of (i, idx, ibp_op, delta_shift, action_prob) actions to
    their respective tasks[i] state via os.fork() over N_WORKERS children.

    Each child gets a contiguous slice of `work` and computes child
    DeltaStates serially. Children inherit master's beam-state memory via
    fork-COW: state.resolved_subs, state.expr, env._raw_eq_cache etc. are
    all available without any IPC. Only the small result tuples are sent
    back via pipes.

    Result order matches what serial would produce (work order = task_idx
    ascending, then action_idx ascending), so downstream candidates.sort()
    + dedup is bit-identical.

    Each result tuple: (action, sol_target, expr, max_w, n_non_masters,
    score). Parent / env / target_sector references are reattached in the
    master process so DeltaState.parent points back to the actual beam
    state object (which a worker fork only had a COW copy of).
    """
    import os as _os
    import pickle as _pickle
    import math as _math

    n = len(work)
    if n == 0:
        return []

    chunk = (n + n_workers - 1) // n_workers
    children = []
    for w_idx in range(n_workers):
        lo = w_idx * chunk
        hi = min(lo + chunk, n)
        if lo >= hi:
            break
        r, w = _os.pipe()
        pid = _os.fork()
        if pid == 0:
            # Child: compute slice [lo:hi], emit results, exit.
            _os.close(r)
            results = []
            for (i, idx, ibp_op, delta_shift, action_prob) in work[lo:hi]:
                state, target, _va, _nt = tasks[i]
                child = _apply_action(state, target, ibp_op, delta_shift,
                                      action_prob, target_sector)
                if child is None:
                    results.append((i, idx, None))
                else:
                    results.append((
                        i, idx,
                        (child.action, child._delta_sol, child.expr,
                         child.max_w, child.n_non_masters, child.score),
                    ))
            try:
                with _os.fdopen(w, 'wb') as f:
                    _pickle.dump(results, f, protocol=_pickle.HIGHEST_PROTOCOL)
            except BrokenPipeError:
                pass
            _os._exit(0)
        else:
            _os.close(w)
            children.append((pid, r))

    # Master: collect from each child in chunk order so result_list is in
    # the same order as the original work list.
    all_results = []
    for (pid, r) in children:
        with _os.fdopen(r, 'rb') as f:
            chunk_results = _pickle.load(f)
        all_results.extend(chunk_results)
        _os.waitpid(pid, 0)

    # Rebuild DeltaState objects in the master's address space so that
    # `parent` refers to the actual beam-state object (the worker had a
    # COW copy but the address-of-Python-object differs).
    candidates = []
    for (i, idx, payload) in all_results:
        if payload is None:
            continue
        action, sol_target, expr, max_w, n_non_masters, score = payload
        state = tasks[i][0]
        env = state._env
        candidates.append(DeltaState(
            parent=state,
            delta_target=action[0],
            delta_sol=sol_target,
            action=action,
            score=score,
            env=env,
            target_sector=target_sector,
            expr=expr,
            max_w=max_w,
            n_non_masters=n_non_masters,
        ))
    return candidates


def _p4_aux_parallel_fork(new_beam, n_workers):
    """Materialize indirect_aux for each survivor in parallel via fork(),
    shipping only the DELTA from parent.aux back to master.

    Worker output per survivor:
        (i, subs, rs, phase_a_changes, phase_b_new_cu_entries,
         phase_b_iraws_meta)

    - phase_a_changes: list of (cu_idx, new_cached, new_ubm) — only the
      entries Phase A's substitution changed (most cu entries pass through
      unchanged when sub_key not in cached, and the new_c IS old_c).
    - phase_b_new_cu_entries: list of (new_cached, new_ubm) for the unique
      new raws Phase B introduced, in insertion order.
    - phase_b_iraws_meta: list of (sub_int, ibp_op, shift) for the new
      iraws entries Phase B introduced (may contain raw duplicates).

    Master reconstructs child aux from parent.aux + delta. Master uses
    its own env.get_raw_equation_cached to materialize raws, so id(raw)
    keys live in master's address space.

    Size: at depth 80, full aux per survivor is ~5 MB; the delta is
    ~0.8 MB. At depth 500+, full aux ~50 MB, delta ~3 MB. The 15-50×
    reduction in IPC payload is what makes parallel P4 viable at depth.
    """
    import os as _os
    import pickle as _pickle
    import time as _time
    from sailir import ibp_env as _env_mod

    _prof = _os.environ.get('DELTA_P4_INSTR') == '1'
    _t_phase = {'fork': 0.0, 'pipe_read': 0.0, 'apply': 0.0,
                'worker_work': 0.0, 'wait_for_first': 0.0}

    n = len(new_beam)
    if n == 0:
        return

    chunk = (n + n_workers - 1) // n_workers
    children = []
    if _prof:
        _t_fork_start = _time.time()
    for w_idx in range(n_workers):
        lo = w_idx * chunk
        hi = min(lo + chunk, n)
        if lo >= hi:
            break
        r, w = _os.pipe()
        pid = _os.fork()
        if pid == 0:
            _os.close(r)
            _worker_t0 = _time.time()
            results = []
            for i in range(lo, hi):
                s = new_beam[i]
                parent_cu, parent_ubm, _parent_rid, parent_iraws = \
                    s.parent.indirect_aux
                n_parent_cu = len(parent_cu)
                n_parent_iraws = len(parent_iraws)
                _ = s.subs
                _ = s.resolved_subs
                _ = s.indirect_cache_list  # forces aux + cache_list
                child_cu, child_ubm, _child_rid, child_iraws = s._indirect_aux

                # Phase A diff: cu entries that changed identity.
                # Incremental code reuses parent dict refs when sub_key not
                # in cached, so identity check correctly distinguishes
                # changed from unchanged.
                phase_a_changes = []
                for j in range(n_parent_cu):
                    if child_cu[j] is not parent_cu[j]:
                        phase_a_changes.append((j, child_cu[j], child_ubm[j]))

                # Phase B: newly-appended cu entries (unique new cacheds).
                phase_b_new_cu_entries = list(zip(
                    child_cu[n_parent_cu:], child_ubm[n_parent_cu:],
                ))

                # Phase B iraws metadata: appended entries beyond parent.
                phase_b_iraws_meta = [
                    (si, op, sh)
                    for (si, op, sh, _raw) in child_iraws[n_parent_iraws:]
                ]
                results.append((
                    i, s._subs, s._resolved_subs,
                    phase_a_changes, phase_b_new_cu_entries,
                    phase_b_iraws_meta,
                ))
            _worker_compute_ms = int((_time.time() - _worker_t0) * 1000)
            try:
                _pickle_t0 = _time.time()
                pickled = _pickle.dumps((_worker_compute_ms, results),
                                        protocol=_pickle.HIGHEST_PROTOCOL)
                _pickle_ms = int((_time.time() - _pickle_t0) * 1000)
                with _os.fdopen(w, 'wb') as f:
                    f.write(pickled)
            except BrokenPipeError:
                pass
            _os._exit(0)
        else:
            _os.close(w)
            children.append((pid, r))
    if _prof:
        _t_phase['fork'] = _time.time() - _t_fork_start

    # Master: read each chunk, apply delta to parent.aux, build child aux.
    env = new_beam[0]._env
    n_indices = _env_mod.N_INDICES
    _wcm_max = 0

    # Concurrent pipe drain. Sequential pickle.load over N pipes serialized
    # both the workers (which block on pipe-write once the kernel buffer
    # fills) and the master (which can only read one pipe at a time).
    # Threads drain all pipes in parallel because read() releases the GIL,
    # so workers can finish writing as soon as their data is consumed and
    # master overlaps reads. This typically cuts P4 pipe_read by ~4-10x at
    # 16 workers.
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _drain(r):
        with _os.fdopen(r, 'rb') as f:
            return _pickle.load(f)

    _first_pipe_t0 = _time.time() if _prof else 0
    with _TPE(max_workers=len(children)) as _ex:
        _drained = list(_ex.map(_drain, [r for (_pid, r) in children]))
    if _prof:
        _t_phase['pipe_read'] = _time.time() - _first_pipe_t0
        # wait_for_first lost meaning under concurrent drain (no "first"
        # blocking read). Report 0 to signal that.
        _t_phase['wait_for_first'] = 0.0

    # Sequential master_apply: cu/ubm/rid/iraws/cache_list builds touch
    # env._raw_eq_cache, which is not thread-safe; per-survivor work is
    # also small enough that the thread overhead would dominate.
    for chunk_idx, (pid, r) in enumerate(children):
        _worker_compute_ms, chunk_results = _drained[chunk_idx]
        if _prof and _worker_compute_ms > _wcm_max:
            _wcm_max = _worker_compute_ms
        if _prof:
            _apply_t0 = _time.time()
        for (i, _subs, _rs, phase_a_changes, phase_b_new_cu_entries,
                phase_b_iraws_meta) in chunk_results:
            s = new_beam[i]
            parent_cu, parent_ubm, parent_rid, parent_iraws = \
                s.parent.indirect_aux

            # Start from parent.aux (shallow copies of the lists).
            child_cu = list(parent_cu)
            child_ubm = list(parent_ubm)
            child_rid = dict(parent_rid)
            child_iraws = list(parent_iraws)

            # Apply Phase A diff.
            for (cu_idx, new_c, new_ub) in phase_a_changes:
                child_cu[cu_idx] = new_c
                child_ubm[cu_idx] = new_ub

            # Apply Phase B: append new unique cu entries + iraws.
            new_cu_iter = iter(phase_b_new_cu_entries)
            for (sub_int, op, sh) in phase_b_iraws_meta:
                seed = tuple(sub_int[k] - sh[k] for k in range(n_indices))
                raw = env.get_raw_equation_cached(op, seed)
                rid_key = id(raw)
                if rid_key not in child_rid:
                    new_c, new_ub = next(new_cu_iter)
                    child_rid[rid_key] = len(child_cu)
                    child_cu.append(new_c)
                    child_ubm.append(new_ub)
                child_iraws.append((sub_int, op, sh, raw))

            # Build indirect_cache_list.
            child_cache_list = []
            for (sub_int, op, sh, raw) in child_iraws:
                idx = child_rid[id(raw)]
                child_cache_list.append((
                    sub_int, op, sh, raw, child_cu[idx], child_ubm[idx],
                ))

            s._subs = _subs
            s._resolved_subs = _rs
            s._indirect_aux = (child_cu, child_ubm, child_rid, child_iraws)
            s._indirect_cache_list = child_cache_list
        if _prof:
            _t_phase['apply'] += _time.time() - _apply_t0
        _os.waitpid(pid, 0)

    if _prof:
        _t_phase['worker_work'] = _wcm_max / 1000.0
        print(f'P4_PARALLEL_INSTR\tn_workers={n_workers} '
              f"fork={_t_phase['fork']:.3f}s "
              f"wait_first={_t_phase['wait_for_first']:.3f}s "
              f"worker_max={_t_phase['worker_work']:.3f}s "
              f"pipe_read={_t_phase['pipe_read']:.3f}s "
              f"master_apply={_t_phase['apply']:.3f}s", flush=True)


def _p1_gv_parallel_fork(p1_work, beam, target_sector, env, filter_mode, n_workers):
    """Run a list of (state_idx, target) gv calls across N forked workers.

    Pre-condition: each beam state's indirect_aux is materialized (workers
    inherit it via COW). Each worker computes per-state filter_subs /
    filter_resolved_subs once (cached within the chunk) and calls
    env.get_valid_actions_with_aux for each (state_idx, target) in its slice.

    Returns a list of (state_idx, target, valid_actions) in the same order
    as `p1_work`, so the downstream tasks list is identical to serial.
    """
    import os as _os
    import pickle as _pickle

    n = len(p1_work)
    if n == 0:
        return []

    chunk = (n + n_workers - 1) // n_workers
    children = []
    for w_idx in range(n_workers):
        lo = w_idx * chunk
        hi = min(lo + chunk, n)
        if lo >= hi:
            break
        r, w = _os.pipe()
        pid = _os.fork()
        if pid == 0:
            _os.close(r)
            state_cache = {}  # state_idx -> (f_subs, f_rs, aux)
            results = []
            for (state_idx, target) in p1_work[lo:hi]:
                entry = state_cache.get(state_idx)
                if entry is None:
                    state = beam[state_idx]
                    if target_sector is not None:
                        f_subs = filter_subs_to_exact_sector(state.subs, target_sector)
                        f_rs = filter_resolved_subs_to_exact_sector(
                            state.resolved_subs, target_sector)
                    else:
                        f_subs = state.subs
                        f_rs = state.resolved_subs
                    indirect_cache = state.indirect_cache_list
                    entry = (f_subs, f_rs, indirect_cache)
                    state_cache[state_idx] = entry
                f_subs, f_rs, indirect_cache = entry
                va = env.get_valid_actions_with_cache(
                    target, indirect_cache, f_subs, f_rs,
                    filter_mode=filter_mode, verbose_timing=False,
                )
                results.append((state_idx, target, va))
            try:
                with _os.fdopen(w, 'wb') as f:
                    _pickle.dump(results, f, protocol=_pickle.HIGHEST_PROTOCOL)
            except BrokenPipeError:
                pass
            _os._exit(0)
        else:
            _os.close(w)
            children.append((pid, r))

    all_results = []
    for (pid, r) in children:
        with _os.fdopen(r, 'rb') as f:
            chunk_results = _pickle.load(f)
        all_results.extend(chunk_results)
        _os.waitpid(pid, 0)
    return all_results


# =============================================================================
# Beam search
# =============================================================================
def beam_search_delta(env, model, start_expr, target_sector,
                     beam_width=40, max_steps=5000, prime=None,
                     verbose=True, stop_on_weight_improvement=True,
                     filter_mode='subsector', device='cpu',
                     checkpoint_path=None, checkpoint_interval=50,
                     checkpoint_time_seconds=300, resume_from=None,
                     checkpoint_mode='thin'):
    """Serial delta-tracking beam search. Matches production semantics for
    --beam-sort weight + --dedup-beam-by-content + DEDUP_VARIANT=rs + NO_TABU.
    Returns (solution, beam, best_weight).

    Checkpointing (matches the pre-delta `beam_search_full.py` pattern):
      - If `checkpoint_path` is given, save the beam atomically every
        `checkpoint_interval` steps OR every `checkpoint_time_seconds`,
        whichever fires first.
      - If `resume_from` points to a previous checkpoint, restore the beam
        and start from the saved step number.
    Survivor aux is converted via FlatAux at save time (strips raw refs;
    re-keyed to env's canonical raws on restore) so id() values stay sane.
    """
    initial_state = DeltaState.root(env, start_expr, target_sector)
    beam = [initial_state]
    best_solution = None
    initial_weight = initial_state.max_w
    best_weight_ever = initial_weight
    resumed_step = 0

    # Resume from checkpoint if requested.
    if resume_from is not None:
        from pathlib import Path as _Path
        import pickle as _ck_pickle
        import gzip as _gzip
        cp = _Path(resume_from)
        if cp.exists():
            # Auto-detect gzipped (thin) vs raw (thick) by magic bytes.
            with open(cp, 'rb') as _f:
                _magic = _f.read(2)
            opener = _gzip.open if _magic == b'\x1f\x8b' else open
            with opener(cp, 'rb') as _f:
                ck = _ck_pickle.load(_f)
            resumed_step = int(ck.get('step', 0))
            best_weight_ever = ck.get('best_weight_ever', best_weight_ever)
            initial_weight = ck.get('initial_weight', initial_weight)
            saved_mode = ck.get('checkpoint_mode', 'thick')
            saved_start = ck.get('start_expr', start_expr)
            beam = []
            if saved_mode == 'thin':
                # Replay each survivor's path from start_expr to rebuild state.
                if verbose:
                    print(f'  Resuming THIN checkpoint: replaying '
                          f'{len(ck["beam"])} survivors x ~{resumed_step} actions...',
                          flush=True)
                t0 = time.time()
                for d in ck['beam']:
                    s = _replay_path(env, target_sector, saved_start,
                                      d['path'], d['score'])
                    beam.append(s)
                if verbose:
                    print(f'  Replay done in {time.time() - t0:.1f}s',
                          flush=True)
            else:
                # Thick: direct restore via FlatAux.
                from sailir.aux_flat import FlatAux
                for d in ck['beam']:
                    flat = d.get('aux_flat')
                    if flat is not None:
                        aux = flat.to_dict_aux(env)
                        cu, ubm, rid, iraws = aux
                        cache_list = []
                        for (sub_int, op, sh, raw) in iraws:
                            idx = rid[id(raw)]
                            cache_list.append((sub_int, op, sh, raw, cu[idx], ubm[idx]))
                        d2 = dict(d)
                        d2['indirect_aux'] = aux
                        d2['indirect_cache_list'] = cache_list
                    else:
                        d2 = d
                    beam.append(DeltaState.from_checkpoint_dict(env, target_sector, d2))
            if verbose:
                print(f'  Resumed from {resume_from} at step {resumed_step}, '
                      f'beam={len(beam)}, mode={saved_mode}, '
                      f'best_weight_ever={best_weight_ever}',
                      flush=True)
        else:
            if verbose:
                print(f'  WARNING: resume_from={resume_from} does not exist; starting fresh.',
                      flush=True)

    # Save closure (uses outer-scope beam, step, best_*).
    # Two modes:
    #   thin  (default) — only paths + score + expr; on resume, each path is
    #                     replayed action-by-action to rebuild the state.
    #                     ~80,000x smaller than thick; right for production
    #                     async reductions at 50k worker scale.
    #   thick           — full FlatAux + subs/RS serialized. Bit-faithful
    #                     resume (no replay). Use for diagnostic single jobs.
    def _save_checkpoint(step_now):
        if not checkpoint_path:
            return
        from pathlib import Path as _Path
        import pickle as _ck_pickle
        import gzip as _gzip
        cp = _Path(checkpoint_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        beam_data = []
        if checkpoint_mode == 'thin':
            for s in beam:
                beam_data.append({
                    'score': float(s.score),
                    'path_len': int(s.path_len),
                    'path': list(s.reconstruct_path()),
                    'max_w': tuple(s.max_w),
                    'n_non_masters': int(s.n_non_masters),
                })
        else:  # thick
            from sailir.aux_flat import FlatAux
            for s in beam:
                _ = s.subs
                _ = s.resolved_subs
                _ = s.indirect_cache_list
                try:
                    flat = FlatAux.from_dict_aux(s.indirect_aux)
                except Exception:
                    flat = None
                beam_data.append({
                    'expr': dict(s.expr) if s.expr is not None else {},
                    'max_w': tuple(s.max_w),
                    'n_non_masters': int(s.n_non_masters),
                    'score': float(s.score),
                    'path_len': int(s.path_len),
                    'path': list(s.reconstruct_path()),
                    'subs': dict(s._subs) if s._subs is not None else {},
                    'resolved_subs': dict(s._resolved_subs)
                                      if s._resolved_subs is not None else {},
                    'aux_flat': flat,
                })
        payload = {
            'checkpoint_mode': checkpoint_mode,
            'step': step_now,
            'best_weight_ever': best_weight_ever,
            'initial_weight': initial_weight,
            'start_expr': dict(start_expr),
            'target_sector': target_sector,
            'beam': beam_data,
        }
        tmp = cp.with_suffix(cp.suffix + '.tmp')
        if checkpoint_mode == 'thin':
            # Gzip the thin checkpoint — at depth 100 it's already KB-sized
            # but gzip drops another 10x via path/expr redundancy.
            with _gzip.open(tmp, 'wb', compresslevel=6) as f:
                _ck_pickle.dump(payload, f, protocol=_ck_pickle.HIGHEST_PROTOCOL)
        else:
            with open(tmp, 'wb') as f:
                _ck_pickle.dump(payload, f, protocol=_ck_pickle.HIGHEST_PROTOCOL)
        tmp.replace(cp)
        # When DELTA_CKPT_EVERY_STEP=1, also save a permanent per-step copy
        # so we can compare beam state step-by-step between two runs.
        import os as _os_ckes
        if _os_ckes.environ.get('DELTA_CKPT_EVERY_STEP', '0') == '1':
            import shutil as _shutil
            per_step = cp.with_name(cp.name + f'.step{step_now:04d}')
            _shutil.copy(cp, per_step)
            if verbose:
                print(f'  Per-step ckpt -> {per_step.name}', flush=True)
        if verbose:
            sz = cp.stat().st_size
            unit = 'KB' if sz < 10*1024*1024 else 'MB'
            sz_h = sz / 1024 if unit == 'KB' else sz / 1024 / 1024
            print(f'  Checkpoint saved at step {step_now} -> {cp} '
                  f'({sz_h:.1f} {unit}, mode={checkpoint_mode})', flush=True)
    last_checkpoint_time = time.time()

    # Action-level tabu (OPT-IN via DELTA_TABU=1, default OFF):
    # tabu[expr_key] = set of (target, ibp_op, delta_shift) tried previously
    # from this expr. Same expression may be revisited later (different sub/RS
    # history), but the same action will never be retried from it.
    # Default OFF so the search behavior remains identical to all baselines
    # (probe_84_*, the running (8,5) reduction) unless explicitly enabled.
    import os as _os_tabu
    _tabu_on = _os_tabu.environ.get('DELTA_TABU', '0') == '1'
    tabu = {} if _tabu_on else None

    # Lazy import to avoid circular load.
    from beam_search import prepare_batched_input_v5

    for step in range(resumed_step, max_steps):
        t_step = time.time()

        # ---- success check ----
        # Production semantics: `solution` is only non-None when an integral is
        # fully reduced to masters (n_non_masters == 0). Stopping on weight
        # improvement returns solution=None — the caller picks the best beam
        # state via `min(final_beam, key=max_weight)`.
        weight_improved = False
        for s in beam:
            if s.max_w < best_weight_ever:
                best_weight_ever = s.max_w
                weight_improved = True
                if verbose:
                    print(f'Step {step}: NEW BEST WEIGHT {best_weight_ever} '
                          f'(started at {initial_weight})', flush=True)
            if s.n_non_masters == 0:
                if best_solution is None or s.path_len < best_solution.path_len:
                    best_solution = s
        if weight_improved and stop_on_weight_improvement:
            if verbose:
                print(f'Step {step}: STOPPING - weight improved', flush=True)
            return best_solution, beam, best_weight_ever
        if best_solution is not None:
            return best_solution, beam, best_weight_ever

        # ---- P1: build tasks + valid actions ----
        # P1_INSTRUMENT=1 emits per-step wall-clock breakdown of P1 phases:
        #   nm_tied : get_non_masters + tied-targets selection
        #   filter  : filter_subs / filter_resolved_subs (per state, once)
        #   aux     : state.indirect_cache_list materialization (lazy first hit)
        #   gv      : env.get_valid_actions_with_cache (per target)
        # DELTA_P1_WORKERS=N forks N workers for the gv-call phase.
        import os as _os_p1
        _p1_instr = _os_p1.environ.get('P1_INSTRUMENT') == '1'
        _p1_t = {'nm_tied': 0.0, 'filter': 0.0, 'aux': 0.0, 'gv': 0.0} if _p1_instr else None
        n_p1_workers = int(_os_p1.environ.get('DELTA_P1_WORKERS', '1'))
        t1 = time.time()
        tasks = []  # list of (state, target, valid_actions, n_tied)

        # Step 1 (master, serial): per-state nm/tied computation.
        # Cheap (~ms total), no IPC concern; just enumerate targets to
        # dispatch.
        state_tied = []  # list of (state_idx, tied_targets)
        for state_idx, state in enumerate(beam):
            non_masters = get_non_masters(state.expr, target_sector)
            if not non_masters:
                continue
            mw = state.max_w
            tied = [k for k in non_masters.keys()
                    if (weight(k)[0], weight(k)[1]) == mw]
            if tied:
                state_tied.append((state_idx, tied))

        # Step 2: gv calls. Build the flat (state_idx, target) work list.
        # Order: state_idx ascending, then target order from tied[]. This
        # matches the serial loop, so downstream tasks order is identical.
        p1_work = []
        for (state_idx, tied) in state_tied:
            for target in tied:
                p1_work.append((state_idx, target))

        # Pre-materialize aux on master so workers' fork-COW snapshot has
        # it (else each worker would re-materialize independently). At
        # end-of-step P4 cleanup we already forced indirect_cache_list, so
        # the property access is cheap here.
        for (state_idx, _) in state_tied:
            _ = beam[state_idx].indirect_cache_list

        if n_p1_workers > 1 and len(p1_work) >= n_p1_workers:
            gv_results = _p1_gv_parallel_fork(
                p1_work, beam, target_sector, env, filter_mode, n_p1_workers,
            )
        else:
            # Serial: cache per-state filter results, exactly like the worker.
            state_cache = {}
            gv_results = []
            for (state_idx, target) in p1_work:
                entry = state_cache.get(state_idx)
                if entry is None:
                    state = beam[state_idx]
                    if target_sector is not None:
                        f_subs = filter_subs_to_exact_sector(state.subs, target_sector)
                        f_rs = filter_resolved_subs_to_exact_sector(
                            state.resolved_subs, target_sector)
                    else:
                        f_subs = state.subs
                        f_rs = state.resolved_subs
                    indirect_cache = state.indirect_cache_list
                    entry = (f_subs, f_rs, indirect_cache)
                    state_cache[state_idx] = entry
                f_subs, f_rs, indirect_cache = entry
                va = env.get_valid_actions_with_cache(
                    target, indirect_cache, f_subs, f_rs,
                    filter_mode=filter_mode, verbose_timing=False,
                )
                gv_results.append((state_idx, target, va))

        # Step 3 (master, serial): assemble tasks list. Each (state_idx,
        # tied_targets) shared n_tied across its gv calls.
        # Action-level tabu (if enabled): filter previously-tried (target,
        # ibp_op, delta_shift) actions out of `va` before scoring. Per-state
        # expr_key is cached so repeated lookups for the same state are O(1).
        n_tied_per_state = {state_idx: len(tied) for (state_idx, tied) in state_tied}
        for (state_idx, target, va) in gv_results:
            if tabu is not None:
                tabu_set = tabu.get(beam[state_idx].expr_key)
                if tabu_set:
                    va = [(op, sh) for (op, sh) in va
                          if (target, op, sh) not in tabu_set]
            if va:
                tasks.append((beam[state_idx], target, va, n_tied_per_state[state_idx]))
        t1_elapsed = time.time() - t1
        if _p1_instr:
            _p1_sum = sum(_p1_t.values())
            print(f'P1_INSTRUMENT\tstep={step}\tt1={t1_elapsed:.4f}\t'
                  f'residual={t1_elapsed - _p1_sum:.4f}\t'
                  f"nm_tied={_p1_t['nm_tied']:.4f}\t"
                  f"filter={_p1_t['filter']:.4f}\t"
                  f"aux={_p1_t['aux']:.4f}\t"
                  f"gv={_p1_t['gv']:.4f}", flush=True)

        if not tasks:
            if verbose:
                print(f'Step {step}: no valid actions, stopping', flush=True)
            break

        # ---- P2: batched model scoring ----
        t2 = time.time()
        batch_data = [(s.expr, s.subs, va, target_sector, tg)
                      for (s, tg, va, _) in tasks]
        batch, n_valid_actions = prepare_batched_input_v5(batch_data, device)
        with torch.no_grad():
            logits, probs = model(
                batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
                batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
                batch['sub_repl_mask'], batch['sub_mask'],
                batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
                batch['sector_mask'], batch['target_integral'],
            )
        t2_elapsed = time.time() - t2

        # ---- P3: build candidates, sort, dedup ----
        t3 = time.time()
        # Build the (task_idx, action_idx, ibp_op, delta_shift, action_prob)
        # work list. task_idx + action_idx are kept as deterministic
        # tie-breakers so a parallel and a serial run produce the same
        # post-sort candidate ordering.
        work = []
        for i, (state, target, valid_actions, n_tied) in enumerate(tasks):
            n_valid = n_valid_actions[i]
            k_per = max(1, beam_width // n_tied)
            top_k = min(k_per, n_valid)
            top_idx = torch.argsort(logits[i, :n_valid], descending=True)[:top_k].tolist()
            for idx in top_idx:
                ibp_op, delta_shift = valid_actions[idx]
                action_prob = probs[i, idx].item()
                work.append((i, idx, ibp_op, delta_shift, action_prob))
                # Record action-level tabu (if enabled). We record every
                # action we choose to TRY here, even if _apply_action later
                # rejects it (sol=None, target absent, etc.) — once we've
                # tried it, we never want to retry.
                if tabu is not None:
                    tabu.setdefault(state.expr_key, set()).add(
                        (target, ibp_op, delta_shift))

        import os as _os_p3w
        n_p3_workers = int(_os_p3w.environ.get('DELTA_P3_WORKERS', '1'))
        if n_p3_workers > 1 and len(work) >= n_p3_workers:
            candidates = _apply_actions_parallel_fork(
                work, tasks, target_sector, n_p3_workers,
            )
        else:
            candidates = []
            for (i, idx, ibp_op, delta_shift, action_prob) in work:
                state, target, _, _ = tasks[i]
                child = _apply_action(state, target, ibp_op, delta_shift,
                                      action_prob, target_sector)
                if child is not None:
                    candidates.append(child)

        candidates.sort(key=_sort_key)

        # Walk in sort order; materialize resolved_subs lazily.
        seen = set()
        survivors = []
        leftovers = []
        for cand in candidates:
            key = _dedup_key_rs(cand)  # triggers RS materialization
            if key in seen:
                leftovers.append(cand)
                continue
            seen.add(key)
            survivors.append(cand)
            if len(survivors) >= beam_width:
                break
        if len(survivors) < beam_width:
            survivors.extend(leftovers[:beam_width - len(survivors)])
        new_beam = survivors[:beam_width]
        t3_elapsed = time.time() - t3

        # ---- P4: eager survivor materialization (incremental aux) ----
        # Force materialization of subs / resolved_subs / indirect_cache_list
        # on the 40 survivors. The cost is dominated by indirect_cache_list
        # which calls compute_indirect_substituted_incremental — the
        # incremental aux update, scaling with |aux| × |affected entries|.
        # This is paid here (not in next step's P1) because we then drop
        # the parent heavy fields to bound memory.
        import os as _os_p4
        _p4_inc_prof = _os_p4.environ.get('BEAM_PROFILE_CIC_INC') == '1'
        n_p4_workers = int(_os_p4.environ.get('DELTA_P4_WORKERS', '1'))
        # Periodic full rebuild of survivor aux (Option R1).
        # Phase A inside incremental update grows cu entry sizes
        # monotonically (each substitution can multiply terms). Over hundreds
        # of steps this is the dominant memory cost (~30-77 GB observed on
        # (8,5) workers at depth 800+). A from-scratch rebuild produces the
        # same algebraic result via `compute_indirect_substituted_with_aux`
        # but with the canonical minimal-size cu dicts, dropping all the
        # incremental Phase A cruft. Default ON, every 50 steps. Set to 0
        # to disable. Trades one expensive step ≈ K incremental steps of
        # CPU for bounded peak memory.
        rebuild_interval = int(_os_p4.environ.get(
            'DELTA_REBUILD_INTERVAL', '50'))
        do_rebuild = (rebuild_interval > 0
                      and (step + 1) % rebuild_interval == 0)
        if _p4_inc_prof:
            ibp_env._cic_inc_reset()
        t4 = time.time()
        t4_subs = 0.0
        t4_rs = 0.0
        t4_aux = 0.0
        t4_rebuild = 0.0
        if do_rebuild:
            # Full rebuild path: subs/RS still need lazy materialization (cheap,
            # COW from parent), then build aux from scratch using current
            # resolved_subs (no parent.aux needed → can immediately drop the
            # parent chain's aux below).
            from sailir.ibp_env import compute_indirect_substituted_with_aux as _full_rebuild
            _sec_proj = _os_p4.environ.get('DELTA_SECTOR_PROJECT_AUX', '0') == '1'
            _ts_proj = target_sector if _sec_proj else None
            for s in new_beam:
                _t = time.time()
                _ = s.subs
                t4_subs += time.time() - _t
                _t = time.time()
                _ = s.resolved_subs
                t4_rs += time.time() - _t
                _t = time.time()
                result, aux = _full_rebuild(
                    s.subs, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
                    target_sector=_ts_proj,
                )
                s._indirect_aux = aux
                s._indirect_cache_list = result
                t4_rebuild += time.time() - _t
            t4_aux = t4_rebuild
        elif n_p4_workers > 1 and len(new_beam) >= n_p4_workers:
            # Parallel fork: each child computes its slice of survivors'
            # aux + cache_list and pickles back. Master attaches to states.
            # Sub-time accounting is lost (workers do subs/rs/aux together);
            # we put it all in t4_aux for now.
            _t = time.time()
            _p4_aux_parallel_fork(new_beam, n_p4_workers)
            t4_aux = time.time() - _t
        else:
            for s in new_beam:
                _t = time.time()
                _ = s.subs
                t4_subs += time.time() - _t
                _t = time.time()
                _ = s.resolved_subs
                t4_rs += time.time() - _t
                _t = time.time()
                _ = s.indirect_cache_list
                t4_aux += time.time() - _t
        # (NOTE) Dead-aux pruning (DELTA_PRUNE_DEAD_AUX) was lossy: it dropped
        # entries whose cached had no overlap with current expr_nm, but a
        # "free integral" in cached can ENTER expr_nm later via a sol_target
        # chain, and the pruned entry would have been the right place to
        # find that action. Removed; the lossless alternative is the
        # expression+resolved-subs anchored rebuild (DELTA_IRAWS_EXPRKEYED).
        # Drop parent heavy fields. parent.action / parent.parent preserved
        # for path reconstruction.
        seen_parents = set()
        for s in new_beam:
            p = s.parent
            if p is None or id(p) in seen_parents:
                continue
            seen_parents.add(id(p))
            p._subs = None
            p._resolved_subs = None
            p._indirect_aux = None
            p._indirect_cache_list = None
            p.expr = None
        t4_elapsed = time.time() - t4

        beam = new_beam

        # ---- Verbose progress ----
        step_total = time.time() - t_step
        if verbose:
            best = min(beam, key=lambda s: s.max_w)
            residual = step_total - t1_elapsed - t2_elapsed - t3_elapsed - t4_elapsed
            _reb_tag = ' REBUILD' if do_rebuild else ''
            print(f'Step {step}: '
                  f'P1(gv)={t1_elapsed:.2f}s '
                  f'P2(model)={t2_elapsed:.2f}s '
                  f'P3(apply+sort+dedup)={t3_elapsed:.2f}s '
                  f'P4(survivor_materialize: subs={t4_subs:.2f}+rs={t4_rs:.2f}+aux={t4_aux:.2f}={t4_elapsed:.2f}s){_reb_tag} '
                  f'residual={residual:.2f}s '
                  f'total={step_total:.2f}s '
                  f'tasks={len(tasks)} cands={len(candidates)} '
                  f'best_max_w={best.max_w} nm={best.n_non_masters}',
                  flush=True)
            if _p4_inc_prof:
                _p = ibp_env._CIC_INC_PROFILE
                print(f'  P4_aux_CIC_INC step={step} n_calls={_p["n_calls"]} '
                      f'avg_prev_cu={_p["len_prev_cu"]//max(1,_p["n_calls"])} '
                      f'avg_prev_iraws={_p["len_prev_iraws"]//max(1,_p["n_calls"])} | '
                      f'phaseA(iter={_p["t_phaseA_iter"]:.3f}s '
                      f'substitute={_p["t_phaseA_substitute"]:.3f}s '
                      f'bitmask={_p["t_phaseA_bitmask"]:.3f}s '
                      f'n_fast={_p["n_phaseA_fast_path"]} '
                      f'n_sub={_p["n_phaseA_substituted"]}) '
                      f'rid_copy={_p["t_rid_copy"]:.3f}s '
                      f'phaseB(subcache={_p["t_phaseB_subcache"]:.3f}s '
                      f'iraws_build={_p["t_phaseB_iraws_build"]:.3f}s '
                      f'dedup_raws={_p["t_phaseB_dedup_raws"]:.3f}s '
                      f'apply_batch={_p["t_phaseB_apply_batch"]:.3f}s '
                      f'new_bitmask={_p["t_phaseB_new_bitmask"]:.3f}s '
                      f'n_new_raws={_p["n_phaseB_new_raws"]} '
                      f'n_unique={_p["n_phaseB_unique_raws"]}) '
                      f'iraws_concat={_p["t_iraws_concat"]:.3f}s '
                      f'result_build={_p["t_result_build"]:.3f}s',
                      flush=True)
            if step < 20:
                nms = get_non_masters(best.expr, target_sector)
                for nm in sorted(nms.keys(),
                                 key=lambda x: (weight(x)[0], weight(x)[1]),
                                 reverse=True)[:5]:
                    print(f'    I{list(nm)} weight={weight(nm)[:2]}', flush=True)

        # ---- Periodic checkpoint ----
        # Trigger on EITHER step-interval (cheap-step runs) OR time-interval
        # (slow-step runs where each step takes many seconds). Atomic via
        # the closure (.tmp + rename).
        if checkpoint_path:
            import os as _os_ckev
            every_step = _os_ckev.environ.get('DELTA_CKPT_EVERY_STEP', '0') == '1'
            now = time.time()
            step_trig = (checkpoint_interval and (step + 1) > 0
                         and (step + 1) % checkpoint_interval == 0)
            time_trig = (checkpoint_time_seconds
                         and now - last_checkpoint_time >= checkpoint_time_seconds)
            if every_step or step_trig or time_trig:
                _save_checkpoint(step + 1)
                last_checkpoint_time = now

    return best_solution, beam, best_weight_ever
