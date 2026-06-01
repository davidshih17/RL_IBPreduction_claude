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
        parent_rs = self.parent.resolved_subs
        new_resolved_sol = apply_resolved_subs(self._delta_sol, parent_rs)
        new_rs = self.resolved_subs  # force materialization
        env = self._env
        result, aux = compute_indirect_substituted_incremental(
            self.parent.indirect_aux, self._delta_target, new_resolved_sol, new_rs,
            env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
        self._indirect_aux = aux
        self._indirect_cache_list = result

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
        return actions

    @property
    def path(self):
        """API-compatibility: production State.path is a list. Lazy build."""
        return self.reconstruct_path()


# =============================================================================
# Helpers
# =============================================================================
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
    _first_pipe_t0 = _time.time() if _prof else 0
    _first_read = True
    for (pid, r) in children:
        if _prof:
            _read_t0 = _time.time()
        with _os.fdopen(r, 'rb') as f:
            _worker_compute_ms, chunk_results = _pickle.load(f)
        if _prof:
            _t_phase['pipe_read'] += _time.time() - _read_t0
            if _first_read:
                _t_phase['wait_for_first'] = _time.time() - _first_pipe_t0
                _first_read = False
            if _worker_compute_ms > _wcm_max:
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
                     filter_mode='subsector', device='cpu'):
    """Serial delta-tracking beam search. Matches production semantics for
    --beam-sort weight + --dedup-beam-by-content + DEDUP_VARIANT=rs + NO_TABU.
    Returns (solution, beam, best_weight).
    """
    initial_state = DeltaState.root(env, start_expr, target_sector)
    beam = [initial_state]
    best_solution = None
    initial_weight = initial_state.max_w
    best_weight_ever = initial_weight

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

    for step in range(max_steps):
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
        n_tied_per_state = {state_idx: len(tied) for (state_idx, tied) in state_tied}
        for (state_idx, target, va) in gv_results:
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
        if _p4_inc_prof:
            ibp_env._cic_inc_reset()
        t4 = time.time()
        t4_subs = 0.0
        t4_rs = 0.0
        t4_aux = 0.0
        if n_p4_workers > 1 and len(new_beam) >= n_p4_workers:
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
            print(f'Step {step}: '
                  f'P1(gv)={t1_elapsed:.2f}s '
                  f'P2(model)={t2_elapsed:.2f}s '
                  f'P3(apply+sort+dedup)={t3_elapsed:.2f}s '
                  f'P4(survivor_materialize: subs={t4_subs:.2f}+rs={t4_rs:.2f}+aux={t4_aux:.2f}={t4_elapsed:.2f}s) '
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

    return best_solution, beam, best_weight_ever
