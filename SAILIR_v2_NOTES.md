# SAILIR v2 — design notes & insights

Running notes capturing architectural decisions, empirical findings, and
gotchas discovered while building/debugging v5 of the beam search (the
strip-passenger / mod-out-by-sub-weight architecture). Intended as
groundwork for SAILIR v2.

Last updated: 2026-06-04.

---

## 1. Architecture: strip-passenger ("active world") state

### Premise
Reducing an integral at starting weight `(w1_s, w2_s)` can be carried out
**inside the quotient** where every integral with weight strictly less than
`(w1_s, w2_s)` is treated as a "passenger" (formally: it lives outside the
active beam state, and reappears only in the final-answer reconstruction
via path replay).

### What v5 strips
- **`expr`**: only target-sector entries with weight ≥ `(w1_s, w2_s)`
  (active non-masters / masters).
- **`resolved_subs` values**: each `RS[K]` value-dict only contains
  active integrals; passenger terms are dropped.
- **`subs` dropped entirely** from `State` — the model only consumes
  `sub_keys` and is insensitive to repl values (see §4).

### What stays
- **Sub-sector passenger content** → routed to `sub_accum` (existing
  Option-F behavior, unchanged).
- **`path`** keeps full `(target, ibp_op, delta)` sequence — used for
  path-replay reconstruction of the full final answer.
- **RS keys** (= past targets) kept in full chronological order — needed
  for iraws enumeration and for dummy subs fed to the model.

### Memory win
Measured on (8,4) Condor probes:
| run                          | drain steps | peak MemoryUsage |
|-----------------------------|-------------|------------------|
| baseline thick (Option F)   | 261         | 13.9 GB          |
| v5 tabu (no incremental)    | 254         |  3.0 GB  (4.6×)  |

Memory is the headline win of v5. Speed parity is the limit (see §7).

### Algebraic correctness (verified)
- **Action set is identical to baseline-full** on the same state. Verified
  on baseline step-166 thick checkpoints: 20/20 + 3/3 fresh re-check
  give `|baseline-full| == |v5-stripped|` bit-equal action sets for every
  (state, target) pair (`scripts/eval/v5_compare_valid_action_sets.py`).
- **Sol's active part is invariant** to whether the substitution chain uses
  full or stripped RS values (mathematical reason: `target` is active, so
  its coefficient in `cached = apply_resolved_subs(raw, RS)` only depends on
  active substitution paths; stripping passenger from RS values doesn't
  change it).
- **Trace verification**: 30-step trace shows `active(expr_v5) ==
  active(expr_full)` and `target_coeff_in_cached_v5 ==
  target_coeff_in_cached_full` at every step
  (`scripts/eval/v5_trace_divergence.py`).

### Reduction completeness
v5 drains the active bucket and yields:
```
start_int = (active masters expansion) + (passenger combination)
```
The passenger combination needs subsequent reductions at lower weight
brackets to reach a pure master expansion. v5 == "one level of weight
reduction"; chain v5 runs at decreasing start weight to get the full
master expansion.

---

## 2. Critical gotchas (don't relearn these!)

### 2.1. Python import of `PRIME`
```python
# BAD:
from sailir.ibp_env import PRIME
# ...later... set_prime(1009)  # updates ibp_env.PRIME but NOT local PRIME!
```
`set_prime()` rebinds the module attribute but doesn't touch already-imported
names. v5 had this bug — every `(coeff * sub_coeff) % PRIME` was actually
modding by 2^31-1 (the default). Coefficients bloated as un-reduced products
(e.g., `1008 * 1008 = 1016064` got stored instead of `1`). The math still
worked mod the real PRIME because intermediate values fit in int64, but
stored representations diverged and the trajectory eventually broke.

**Fix:** `import sailir.ibp_env as ibp_env` and use `ibp_env.PRIME` for every
modular arithmetic op. Run a sanity check after each step that all stored
coefficients are `< PRIME`.

### 2.2. Best-state tracker by score
Tracking "best state seen so far" by cumulative `−Σ log(p)` is broken: the
initial state has score=0, and every later state has negative score, so the
initial wins every comparison.

**Fix:** track by progress key `(max_w, n_non_masters, -len(path))`. Tie-break
on path length to prefer states that made it further.

### 2.3. Dummy subs for the model
The classifier is **insensitive to the `sub_repl_ints` / `sub_repl_coeffs`
values** (variants B/C/D in `scripts/eval/v5_test_dummy_subs.py` give
|logit Δ| ≤ 1.5e-5 versus baseline). But:
- It IS sensitive to the **number** of sub_keys: variant E (empty subs,
  all positions masked) drifts by ~1 in logit, ~0.05 in prob.
- The right dummy: keep the `sub_keys` slots populated with **real RS keys**
  (variant F); set `sub_repl_ints`/`sub_repl_coeffs`/`sub_repl_mask` to
  zeros.

This is what `prepare_batched_input_v5_dummy` does.

---

## 3. Action enumeration (v5 == training)

`enumerate_valid_actions_with_indirect_cache(target, indirect_cache, subs,
resolved_subs, …)` is the same code path baseline and training use. v5 does
not modify it.

Two phases:
- **Phase 1a (direct)**: anchor = target. For each `(op, shift)` in topology,
  `seed = target − shift`, check `raw` has target with nonzero coeff and
  `cached = apply_resolved_subs(raw, RS)` still does too. Bounded by
  topology size (~30-60 actions per target).
- **Phase 1b (indirect)**: anchor = past sub_int K. iraws contains tuples
  `(sub_int, op, shift, raw, cached, ub)` produced by
  `compute_indirect_substituted_*`. For each, check `target ∈ cached`.
  Action emitted: `(op, delta = sub_int − shift − target)`.

Empirical breakdown on 254-step v5 tabu drain (`v5_study_action_anchors.py`):
- Phase 1a only: 33.1%
- Phase 1b only: 14.2%
- Phase 1a OR 1b (Phase 1a fires first; 1b redundant): 52.8%
- → **86% of chosen actions are reachable via Phase 1a alone**

---

## 4. iraws pruning: the "first-N anchor" insight

**Key empirical finding (2026-06-04)**: On the 254-step (8,4) drain of v5
tabu, ALL 36 Phase-1b-only chosen actions used anchors from sub_ints added
in the **first 25 steps** (median insertion step = ~5; max = 25).
Late-drain steps (250–253) used anchors aged 226–239, i.e., sub_ints added
at steps 11–28.

**Recency-based pruning is exactly backwards.** A `--iraws-window=50`
(keep last 50 sub_ints) would have killed 100% of late-drain hooks and
left the model wandering at high weight indefinitely (verified — Condor
1544052 was stuck at mw=(9,4) for 100+ steps).

**Correct pruning**: `--iraws-keep-first N` (keep iraws anchored on the
FIRST N sub_ints, where N ≈ 50 has empirically captured 100% of needed
anchors with comfortable margin).

### Why the first N sub_ints are special — structural reason
The first ~25 sub_ints of any reduction of `start_int` are
**lattice-neighbors** of `start_int` (differ by ±1–2 in 1–4 index positions).
This is because:

1. Each step's target = max-weight non-master in expr.
2. Early in the path, the only non-masters are those produced by direct
   IBP on `start_int` itself. Those sols' active-part integrals all sit in
   a small ball around `start_int` in the lattice.
3. So the first ~25 targets (= first ~25 sub_ints) are systematically
   close to `start_int`.

iraws for sub_int K cover integrals reachable by one topology shift from K.
So early sub_ints' iraws cover integrals within ~2-3 shifts of `start_int`
— which is **exactly the frontier the active-weight reduction traverses**.

Later sub_ints are products-of-products, drift farther in the lattice, and
their iraws cover specialized regions that rarely contain the model's
high-weight targets.

**Prediction**: `--iraws-keep-first N` with N tuned to the topology's
neighborhood size around `start_int` should generalize to other starting
integrals. It's a topology-driven hyperparameter, not path-specific.

### Anchor-usage frequency (254-step v5 tabu drain)
Just 6 sub_ints accounted for ~80% of all Phase-1b-only anchor uses:
```
sub_int @ step  2:  6 uses   sub_int @ step 11:  5 uses
sub_int @ step  3:  7 uses   sub_int @ step 12:  3 uses
sub_int @ step  6:  3 uses
sub_int @ step  8:  4 uses
```
These are the "central hubs" in `start_int`'s lattice neighborhood.

---

## 5. Tabu (per-expr action blocklist)

Implemented as `--tabu` flag (matches `delta_beam_search.py:DELTA_TABU=1`
semantics): for each expr fingerprint, record `(target, op, delta)` tuples
already chosen from this expr, then filter them out next time the same expr
is encountered.

### Why tabu matters in v5
Without tabu, v5 cycles. Empirically: 24 of 40 beam slots all converge to
the same expr (different RS histories), starving exploration.
Tabu forces the beam to explore alternative actions per expr.

The v5 tabu run drained (8,4) in 254 steps; without tabu it didn't drain
in 30+ steps with the same per-step pattern (mw=(8,4), nm=1, cycling).

### Tabu fingerprint
`frozenset(expr.items())`. Acceptable to recompute per use; the expr is
small in v5 (1-3 entries) so this is cheap. If needed, can cache
`expr_key` on `State_v5` (v4 does this).

### Tabu MUST be checkpointed for bit-identical resume (2026-06-05)
The very first bit-identical-resume attempt diverged at step 7 — diff
of valid action lists showed scratch ⊊ resumed (resumed had +20 actions
per task on 32/80 tasks). Trace: scratch accumulated tabu entries from
steps 1-5; resumed initialized `tabu_dict = {}` at line 584 and never
restored. So at step 6, resumed re-emitted ~20 already-tried actions
per task that scratch had blocked.

Fix (`beam_search_v5.py`):
1. `_serialize_tabu_for_ckpt()` converts the dict to a list of
   `(tuple(expr_items), [(target, op, delta), …])` pairs.
2. Saved alongside `beam` in both rolling and per-step ckpts.
3. On resume: walk `d['tabu_dict']`, rebuild `{frozenset(expr_items):
   set(entries)}`.

After fix: `valid-SET match: 80/80` at step 7 from `.step0006` ckpt
(`logs/v5_dump_valids_diag_1780632194.log`).

Diagnostic recipe — `scripts/eval/v5_dump_valids_diag.sh`:
* `V5_DUMP_VALIDS_AT_STEP=N` env var pickles every `(parent_idx,
  target, tuple(valid))` triple at step N — drop into both runs, diff
  set-wise per task.

Related: `start_w12` IS saved in the ckpt (line 910/922) but the
resume loader **does not read it** — it relies on the CLI passing the
same `--integral`. If a future caller resumes with a different integral
it will silently use a different `start_w12`. Consider asserting
`d['start_w12'] == start_w12` on resume.

---

## 6. Anti-pattern: `compute_indirect_substituted_exprkeyed_delta` ("rabbit hole")

The bounded-anchor exprkeyed function CLAIMS bit-identical action set vs
depth-keyed baseline (docstring). In practice:
- Action SET is approximately the same.
- iraws **ORDER** differs from depth-keyed.
- Under `max_actions` truncation, the model sees a different first-900 →
  different softmax denominator → tiny prob drift → diverging trajectory
  by step 4 (`scripts/eval/v5_verify_exprkeyed.sh`).
- Score diverged by 0.288 (not FP noise) at step 6. Path[3] differed
  between depth and exprkeyed runs — entirely different actions chosen.

**Don't switch to exprkeyed-delta as a "lossless drop-in".** Same rabbit
hole as v4-rescue vs baseline. If you use it, commit to it as a NEW
reference trajectory (which then needs its own verification).

---

## 7. Profiling: where the time actually goes

Late-stage (step 252, |RS|=251, v5 tabu, no pruning) breakdown:
```
t_step = 26.5s
  P1 (build tasks+valid):   5.2s  (nm_tied + aux + gv + tabu)
  P2 (model forward):       4.6s  (≈ batch_size × max_actions)
  P3 (apply + sort):       11.9s  (apply=11.89, sort=0.05)
  P4 (attach_aux survivors):4.7s
```

Top bottlenecks (and what bites them):
1. **`apply` in P3 (45%)** — 1600 candidates × ~7ms per
   `apply_resolved_subs(raw, RS) + add_sub_to_resolved_v5(...)`. Scales
   with |RS|. Not addressable without algorithmic change.
2. **`model_fwd` (17%)** — transformer over fixed `(batch × 900-ish action)`
   tensor. Insensitive to RS stripping. CPU-bound.
3. **`attach_aux` (18%)** — 40 survivors × incremental Phase A over
   cu_size ≈ 36k entries. Bounded if iraws is bounded
   (e.g., `keep-first=N` → cu drops ~5×).
4. **`gv` + `aux_repack` (~14%)** — also benefits from bounded iraws.

### Cheap lossless wins applied
- **Cache `max_w12` / `total_w12` on `State_v5`** — sort key becomes O(1)
  lookup. Sort time dropped from 22% of step → ~0.2% (5s → 0.05s at step
  252). Just add the fields at `State_v5` construction.
- **Incremental aux update via
  `compute_indirect_substituted_incremental`** (depth-keyed,
  bit-identical to fresh rebuild — verified 10/10 steps). Modest at
  small |RS|; meaningful at high |RS|.
- **LAZY_RS** (`--lazy-rs`, default ON). Defer `add_sub_to_resolved_v5`
  from every candidate to survivors only. `apply_action_v5` returns
  `(child, sol)` with `child.resolved_subs=None`; `_materialize_lazy_rs`
  runs post-selection. Bit-identical (8/8 steps verified). **2.2× wall
  speedup at step 252 (18.8s → 8.1s); P3 apply went 12.1s → 1.3s
  (9.6× shrink, since 1600 add_sub calls collapsed to 40).** Cumulative
  with `--iraws-keep-first=50`, v5 is ~4× faster than original tabu at
  depth.
- **Per-step thick checkpoint** (`--ckpt-every-step`) for bit-identical
  verification across two runs (`scripts/eval/v5_diff_step_ckpts.py`).

### Cheap LOSSY but principled wins
- **`--iraws-keep-first 50`** (see §4) — shrinks iraws ~5× at high |RS|
  while preserving the empirically-critical "bootstrap anchors". Speeds
  up gv + attach_aux + aux_repack. Doesn't help `apply`.

---

## 8. State_v5 schema reference

```python
State_v5 = namedtuple('State_v5', [
    'expr',           # target-sector dict, ACTIVE-weight only
    'resolved_subs',  # past targets → resolved sol (values stripped of passenger)
    'sub_accum',      # sub-sector passenger (Option F, unchanged)
    'score',          # cumulative log(action_prob)
    'path',           # list of (target, ibp_op, delta) for replay
    'n_non_masters',  # cached count for sort key
    'max_w12',        # cached max (w1,w2) of non-masters for sort key
    'total_w12',      # cached sum (w1,w2) of non-masters for sort key
    'aux_flat',       # (cu, ubm, rid, iraws) or None to mean "lazy rebuild"
])
```

**Not in State (vs prior `beam_search_full.py`'s `State`):**
- `subs` — dropped; model insensitive to values, RS keys serve as dummy
- `indirect_aux` / `indirect_cache_list` — folded into `aux_flat`

---

## 9. Path replay for final answer

`replay_full_expr(start_expr, path, env)` in `beam_search_v5.py` replays
the `path` against the unstripped `start_expr` using
`apply_all_substitutions` (full algebra, no stripping). Recovers the full
final expression including all passenger spillover.

Used in `--output` to write the post-reduction full expr alongside
`best_state.path`.

Verification: `scripts/eval/v5_verify_replay.py` replays a saved path and
reports the max-active-weight of remaining non-masters per step. For the
v5 tabu (8,4) drain, replay succeeds at every step and ends with 0 active
non-masters + 6 passenger non-masters at (8,3)/(7,4)/(7,3) (each needs
its own subsequent reduction).

---

## 10. Verification tooling worth keeping for v2

| script                              | purpose                                  |
|-------------------------------------|------------------------------------------|
| `v5_compare_valid_action_sets.py`   | head-to-head action set: baseline vs v5  |
| `v5_trace_divergence.py`            | per-step active-state equality check     |
| `v5_diff_step_ckpts.py`             | per-step thick-ckpt diff (bit-identical) |
| `v5_verify_replay.py`               | path replay correctness                  |
| `v5_test_dummy_subs.py`             | model sensitivity to subs values         |
| `v5_study_action_anchors.py`        | per-step phase + anchor-age classifier   |
| `v5_plot_anchor_ages.py`            | anchor-age scatter visualization         |
| `v5_compare_step1.py`               | step-1 algebra cross-check               |
| `v5_check_cycling.py`               | expr fingerprint repeats within path     |

---

## 11. Open questions / next directions

1. **Bound `apply` cost.** Per-child `apply_resolved_subs + add_sub_to_resolved_v5`
   is 45% of step time at high |RS|. Cython for `add_sub_to_resolved` was
   attempted in legacy `_add_sub_inner.so` (`apply_sub_inner`) but not
   wired in current code path. Worth profiling end-to-end.
2. **`--iraws-keep-first N` topology dependence.** Currently N≈50 works
   for pentagonbox (8,4). Need to characterize N as a function of
   `(topology, start_w)` — is it the lattice-neighborhood size?
3. **Hybrid pruning**: `--iraws-keep-first M --iraws-window K` keeps both
   ends; might be even safer.
4. **Model retraining on stripped states.** Current model trained on
   full-expr baseline. v5's compact expr is OOD for it. Retraining on
   v5-style stripped trajectories should give a model that ranks
   appropriately within the active world (may also reduce reliance on
   late-anchor Phase-1b hooks).
5. **Chained brackets.** Once v5 drains the active bucket, the passenger
   non-masters need their own reductions at lower start_w. Automate the
   chain: when v5 finishes, spawn sub-reductions on each remaining
   passenger integral.

---

## 12. File map

Live code:
- `scripts/eval/beam_search_v5.py` — the v5 beam search and CLI
- `scripts/eval/probe_84_v5_tabu.sh` — sample condor submit
- `scripts/eval/probe_84_v5_tabu_first50.sh` — with first-50 anchor keep

Design docs:
- `DESIGN_v5_iraws.md` — original v5 design proposal (pre-implementation)
- `SAILIR_v2_NOTES.md` — this file

Verification:
- `scripts/eval/v5_*.py` family (see §10 table)
