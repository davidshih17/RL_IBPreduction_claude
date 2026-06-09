# SAILIR v6 / hier orchestrator — known issues

Running list of bugs / rough edges to address. Order is rough priority,
not strict.

## 1. `apply_substitutions` 10,000-iter cap fires on failed-worker identity entries

**Where:** `scripts/eval/hierarchical_reduction.py:49-80`
([apply_substitutions](scripts/eval/hierarchical_reduction.py#L49-L80))

**Symptom:** Repeated `WARNING: apply_substitutions exceeded 10000 iterations`
in the orchestrator log, one per iter, starting from when the first
failed-worker pickle is processed.

**Cause:** When a worker reports `success=False`, the orchestrator caches
the integral as `cache[X] = {X: 1}` (identity self-mapping; line 826 in
runtime path, 472 in resume path) so the orchestrator doesn't keep
re-submitting it. The `apply_substitutions` loop sets `changed = True`
whenever it touches a cache entry — but for an identity mapping the
substitution puts X right back. So every iteration changes nothing
visible but loops forever until the 10k safety cap.

**Impact:** *Correctness* — final reduction is incomplete; the
un-reducible integral stays in the expr. *Performance* — every
orchestrator iter pays 10k × |expr| useless dict ops.

**Fix (one-liner):**
```python
if integral in cache and cache[integral] != {integral: 1}:
    # do the substitution
else:
    new_expr[integral] = ...   # treat identity as "no substitution"
```
Or detect identity at cache-write time and store the sentinel `None`
(skip in the loop).

**Notes:** First seen on the (8,5) v6 sweep, 2026-06-07, where ~6 workers
hit a tabu trap and got cached as identity. Per-iter cost in that run
was tolerable (orchestrator RSS flat) but on larger sweeps the
performance hit will dominate.

---

## 2. `query_job_start_times` 30-second timeout fires on large queues — ✅ FIXED 2026-06-08

**Where:** `scripts/eval/hierarchical_reduction.py:253-283`
([query_job_start_times](scripts/eval/hierarchical_reduction.py#L253-L283))

**Symptom:** `Error querying JobStartDate: Command '['condor_q',
'<cluster_id_1>', ..., '<cluster_id_N>', '-af', 'ClusterId',
'JobStartDate']' timed out after 30 seconds` appears in the orchestrator
log when the queue holds several thousand pending Condor jobs (~74×
during the (8,5) sweep).

**Cause:** The straggler-detection probe issues a single
`condor_q <all_pending_cluster_ids> -af ClusterId JobStartDate` to the
schedd. With ~9k+ clusters the schedd takes longer than 30 s to walk and
return the result.

**Impact (turned out non-benign):** When the query timed out, the
returned `{}` silently broke the running-worker protection in the
obsolete-integral kill path. Every "obsolete" worker was treated as
idle, so the orchestrator condor_rm'd RUNNING workers too — accounting
for ~155 of the 157 NO_RESULT_NO_ERR failures on the (8,5) sweep
(workers killed mid-progress, sometimes after 88 min / 130 v6 steps).

**Fix applied 2026-06-08:** Removed the `query_job_start_times` call
from the obsolete-cancel path entirely. Policy is now "always kill
pending jobs (idle OR running) when their integral leaves expr" —
consistent across all iters, no JobStartDate timeout to worry about.
See [hierarchical_reduction.py#L514-L539](scripts/eval/hierarchical_reduction.py#L514-L539).

The function `query_job_start_times` is still defined (it's also used
by the straggler-escalation path), so this fix only addresses the
obsolete-cancel timeout. If straggler escalation is ever re-enabled,
the 30 s timeout on a 9k-cluster query will still be a problem and
the original batching/raise-timeout fix options below remain valid.

**Original fix options (still apply to straggler path):**
- **Batch** the query into chunks of ~1,000 cluster IDs per call, merge
  results.
- **Raise timeout** to 120 s.
- **Skip entirely** when both straggler timeouts are at the sentinel
  large-value (≥ 1e9 seconds), since the result would be unused.

---

## 3. v6 stalls on some integrals: beam → 1 expr + tabu blocks all

**Where:** `scripts/eval/beam_search_v6.py` (action enumeration + tabu
filter + macro-dedup).

**Symptom:** v6 worker prints `[v6 step N] no tasks — STUCK` and exits
INCOMPLETE. result.pkl has `success=False`, `best_n_non_masters >= 1`,
`best_max_w12` somewhere above the masters tier.

**Cause:** After macro-dedup collapses the beam to `uniq_expr=1` (one
distinct expression), tabu has accumulated entries for every
`(target, op, delta[, eq_fp])` the model already picked from that
fingerprint. The Phase 1a + Phase 1b enumeration then comes up empty
after the tabu filter. v6's main loop has no fallback — it just
declares no tasks and exits.

**Impact:** Each stall produces one failed pickle. On the (8,5) sweep,
6 of ~7,300 workers hit this — 0.08 % failure rate, vs 1,941 failures
(estimated ≈ 50 %) on the prior delta sweep. So it's a real problem
but a small one.

**Fix ideas:**
- **Clear tabu and retry** when the beam collapses (worker-level
  recovery).
- **Detect "tabu trap"** before stalling: if every valid action is in
  tabu_set for the surviving expr, drop the OLDEST tabu entries for
  that expr_fp and continue.
- **Loosen iraws-keep-first** mid-run: if a worker can't make progress
  with `iraws_keep_first=50`, fall back to no truncation.
- **Round-2 hierarchy:** track failures in `failed_integrals.txt` and
  resubmit with a different config (`--no-tabu` or larger
  `--iraws-keep-first`). Already in progress for the (8,5) sweep.

---

## 4. Stale running workers can't be cancelled

**Where:** orchestrator's "cancel pending jobs whose integrals are no
longer needed" path
([hierarchical_reduction.py — search "Cancelled N idle pending jobs"](scripts/eval/hierarchical_reduction.py)).

**Symptom:** Sometimes a `Completed I[...]` line shows up for an
integral that no longer appears in the current iter's `[hist]` /
`[maxw]`. The worker had been launched in an earlier iter when the
integral was still in the expr; by the time the worker finished,
intermediate cache chains had substituted that integral out via other
worker results. The completion is "stale" — its result.pkl goes into
the cache for that specific integral but doesn't move the global expr
forward.

**Cause:** Condor distinguishes idle jobs (cancellable via
`condor_rm`) from running jobs (can be killed but the result is lost).
The orchestrator currently only `condor_rm`s idle jobs whose integrals
are no longer needed; running workers complete naturally even if their
result won't be used.

**Impact:** *Cluster waste* — the running stale worker may run for
hundreds of steps doing genuinely useful but redundant work. On the
(8,5) sweep one example: 183-step / 600+ s worker on a stale L7 r=8
integral, finishing well after its integral had been substituted via
shorter chains.

**Fix ideas:**
- **`condor_rm -fast` running workers** whose integrals are no longer
  needed. Worker's partial output is lost but their slot is freed.
- **Soft cancel:** orchestrator writes a sentinel file the worker
  polls; worker exits early when it sees the sentinel.
- Accept the waste — it's typically a single-digit-percent overhead
  because the orchestrator iter cadence keeps up with worker turnover.

---

## 5. iraws-keep-first 50 is topology- and depth-dependent

**Where:** `scripts/eval/beam_search_v6.py` `--iraws-keep-first` arg.

**Context:** Empirically `N=50` works for pentagonbox (8,4) and the
canonical long-runner. But the "first-N anchor" insight is rooted in
the lattice-neighborhood size around the start integral, which depends
on topology and start_w12. Bigger / less-canonical starts may need a
larger N. Currently no per-(topology, start_w12) tuning is automated.

**Fix idea:** Auto-tune N as `(empirical 80 % anchor coverage) * 1.5`
during a short warm-up phase of each worker. Or just hard-code a
mapping `(topology, w_class) → N`.

---

## 6. v6 worker leaves rolling checkpoint files on disk after success

**Where:** `scripts/eval/onestep_worker_v6.py` (end of `main()` after
writing `result.pkl`).

**Symptom:** After a long sweep, the work-results directory accumulates
hundreds of GB of stale `*.pkl.checkpoint` files belonging to workers
that have already written their final `result.pkl`. On the (8,5) sweep
at the 33,910-result mark, **5,103 leftover checkpoints from finished
workers** consumed ~360 GB (mean 72.5 MB, max 633 MB).

**Cause:** `beam_search_v5.beam_search_v5()` writes a rolling checkpoint
at `ckpt_path` every `ckpt_every` steps (default 50) via an atomic-rename
into place. The worker's wrapper (`onestep_worker_v6.main()`) writes the
final `result.pkl` when the search finishes — but the checkpoint file is
**never deleted**, so the last write to `ckpt_path` sits on disk
permanently.

**Impact:** *Disk waste*, growing roughly linearly with number of
workers. The orchestrator never reads checkpoints from successful
workers (it cares about `result.pkl`), so they're pure dead weight. Did
not break the (8,5) sweep — `/het/p4` has 6.9 TB free against the
~400 GB pile-up.

**Fix:**
```python
# at the end of onestep_worker_v6.main(), after result.pkl is written
if args.checkpoint_path and not args.no_checkpoint:
    for p in (args.checkpoint_path,
              args.checkpoint_path + '.tmp'):
        try:
            os.remove(p)
        except OSError:
            pass
```
And for full hygiene, also delete sibling per-step thick checkpoints
(`<ckpt_path>.stepNNNN`) if `--ckpt-every-step` was on.

**One-time cleanup for an in-flight sweep** (safe — only touches
checkpoints whose result.pkl is already on disk with `success=True`):
```bash
python -c "
import os, pickle
d = 'results/<sweep>/work/results'
for fn in os.listdir(d):
    if not fn.endswith('.pkl.checkpoint'): continue
    parent = os.path.join(d, fn[:-len('.checkpoint')])
    if not os.path.exists(parent): continue
    try:
        if pickle.load(open(parent,'rb')).get('success'):
            os.remove(os.path.join(d, fn))
    except Exception: pass
"
```

---

## 7. Per-iter `[hist]` print is silently truncated at top-15 buckets

**Where:** `scripts/eval/hierarchical_reduction.py:862-863`
([print site](scripts/eval/hierarchical_reduction.py#L862-L863))

**Symptom:** The `[hist] L8r8:N L8r7:N …` line printed after every iter
sums to noticeably less than the `non_masters` count in the matching
iter line. Example from (8,5) sweep iter 544: iter says
`19290 non-masters` but the `[hist]` entries sum to only 6,292. The
missing 13,000 entries are hidden silently — the user has no way to tell
that L3r<5 (8,817 entries) and ALL of L2 (4,181 entries) were dropped
from the print.

**Cause:**
```python
top = sorted(Lr_counts.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))[:15]
hist_str = " ".join(f"L{L}r{r}:{n}" for (L, r), n in top)
```
The hardcoded `[:15]` cap keeps the line short, but for a deep sweep
where the tail of low-`r` buckets at low `L` carries thousands of
integrals, those get cut off.

**Impact:** *Operator confusion* — the print misrepresents the active
expression. The `[maxw]` line below still gives the correct per-level
totals so this is a readability bug, not a correctness one.

**Fix:** Remove the cap entirely so every non-empty `(L, r)` bucket is
shown:
```python
top = sorted(Lr_counts.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))
hist_str = " ".join(f"L{L}r{r}:{n}" for (L, r), n in top)
```
(If line length becomes a problem for very wide sweeps, prefer wrapping
across multiple print lines rather than re-introducing a silent cap.)

---

## 8. OOM audit gap — PyTorch `RuntimeError` allocation failures go uncounted

**Where:** Audit tooling — not a code bug per se, but a serious blind
spot in how we were reporting sweep health. The relevant Condor logs +
worker .err files are at
`results/<sweep>/work/logs/async_*.{log,err,out}`.

**Symptom:** During the (8,5) v6 sweep I repeatedly claimed "zero OOMs"
across all 13k–63k completed workers, based on:
  - `condor_q dshih -nobatch | grep H` → 0 held jobs.
  - grep across `.err` for `MemoryError|killed|OOM|condor_rm`.
  - Condor event log scan for `Job was held / SYSTEM_PERIODIC_REMOVE /
    Image size of job updated: ≥ 16777216 KB / terminated by signal`.

  All three came up empty. But ~305+ workers had actually OOM'd: PyTorch
  raises `RuntimeError: Cannot allocate memory ... alloc_cpu.cpp` from
  inside `multi_head_attention_forward` when the CPU allocator can't
  grow further. The worker exits with return code 1, Condor logs
  `Normal termination (return value 1)`, no result.pkl is written, and
  none of my detection patterns matched.

**Why my audit missed it:**
  1. **Wrong exception name**. PyTorch uses `RuntimeError`, not
     `MemoryError`. Pure-Python OOMs raise `MemoryError`, but PyTorch's
     C++ allocator throws via `[enforce fail at alloc_cpu.cpp:124]`.
     The strings to grep are `Cannot allocate memory` or
     `alloc_cpu.cpp`, not `MemoryError`.
  2. **Wrong Condor signal**. The job exits cleanly from Condor's
     perspective. It's NOT held, NOT removed, NOT signaled — the
     Python process returns 1 normally. The "Job was held" /
     "terminated by signal" patterns never match.
  3. **`request_memory` is a soft hint, not a cap** in our cluster.
     Workers happily run at 2–3× their request before hitting alloc
     pressure, so "Image size > request" doesn't fire as a kill event.
  4. **`success=False` result.pkl ≠ failure**. I was equating these,
     missing the case where there's NO result.pkl at all (the worker
     crashed before reaching its final-output write).

**Impact:** Real failure rate on (8,5) sweep so far is ≈ **610 / 63,435
≈ 1.0 %** (305 OOMs + 263 tabu-traps + 2 non-OOM tracebacks + 40
"other"). I was reporting **263 / 63,435 ≈ 0.4 %**.  Still far better
than the delta sweep's ~1,941 OOMs, but ~3× higher than the headline
number I was using.

**Fix (audit-tooling level):**
  - **Detection patterns** that catch PyTorch OOMs:
    - in `.err`: `Cannot allocate memory`, `alloc_cpu.cpp`, `bad_alloc`,
      `OutOfMemory`, `MemoryError`.
    - workers with `.log` ending in `Job terminated` but NO matching
      result.pkl → categorize separately as
      `crashed-before-result`.
  - **Build a per-failure-mode summary** instead of one all-up count.
    `scripts/eval/build_failed_integrals.py` (added 2026-06-08) now does
    this, producing
    `<sweep>/failed_integrals.csv` (full) plus
    `<sweep>/failed_integrals_<CATEGORY>.txt` (per-category lists for
    round-2 dispatch).
  - **Cross-check against `condor_history` `ExitCode != 0`** — those are
    Python-level exits (code != 0) and complement the result.pkl
    presence check.

**Fix (worker-level, helps both prevention and detection):**
  - Wrap the v6 worker main in a `try/except` that on any unhandled
    exception writes a sentinel result.pkl with `success=False` + the
    traceback. That way EVERY worker that runs to termination produces
    a result.pkl, and "no result.pkl" cleanly means "still running" —
    no inference about why it disappeared.
  - Even simpler: have the orchestrator write a per-pending-integral
    "started at iter N" record, so the post-hoc audit can compare
    "started but never returned a result.pkl" workers vs Condor's
    notion of completed clusters.

---

## 9. Orphaned pending workers — crashes without `result.pkl` are never reaped

**Where:** orchestrator completion-detection loop
([hierarchical_reduction.py:770-832](scripts/eval/hierarchical_reduction.py#L770)).

**Symptom:** OOM victims, MKLDNN tracebacks, silent SIGKILLs, and now
manual `condor_rm`'d hog workers (anything that dies without writing
`result.pkl`) get stuck in `pending` forever. Their integral stays in
`expr`, contributing to every `[hist]` bucket it occupies, and the
orchestrator will not resubmit it.

**Cause:** Completion is detected only via `output_file.exists()`
(line 773). The `check_job_status` helper is defined at line 286 but
never called — there is no "cluster vanished without a result.pkl"
recovery path.

**Impact:** Currently ~413 crashed workers across OOM_PYTORCH +
OOM_OTHER + OOM_SILENT_SIGKILL + TRACEBACK + MANUAL_OOM. Their
integrals are parked in `expr`, the orchestrator can never reach 0
non-masters, and `[hist]` becomes increasingly dominated by these
permanently-stuck entries as the sweep proceeds.

**Fix ideas:**
- Periodic `check_job_status(pending_cluster_ids)`: any cluster_id no
  longer in the queue AND no `result.pkl` ⇒ write a sentinel
  `result.pkl` with `success=False`, `original_integral=integral`,
  `final_expr={integral: 1}`. Orchestrator then identity-caches it via
  the existing failed-worker branch. Integral stays in expr (correct)
  but the worker exits pending so subsequent retries can be scheduled.
- Wrap worker `main()` in `try/except` that on any unhandled exception
  writes a sentinel result.pkl. Then "no result.pkl" cleanly means
  "still running" — no inference required.

---

## 10. v6 worker memory bloat: aux_flat duplicated 40× across beam + sub_accum dead + tabu uncapped

**Where:** `scripts/eval/beam_search_v6.py` State_v5 layout, lines
180-188; `_prune_aux_by_recency` line 140; aux build sites in
`sailir/ibp_env.py` (compute_indirect_substituted_*).

**Symptom:** On the (8,5) sweep 60 workers hit ≥8 GB RSS, with two at
19.5 GB / 17.1 GB after 22.9 h. All clustered at a 9,766 MB plateau —
deterministic in step count, not integral-specific. The v6 design was
supposed to cap this via `iraws_keep_first=50` + LAZY_RS + macro-dedup.

**Cause:** Three independent contributors:

1. **`aux_flat` is duplicated across all 40 beam slots.** At step 1550
   every slot has identical `len(iraws)=11,548`, `len(cu)=8,734`, and
   `cu_bytes≈14.5 MB`. `iraws_keep_first=50` caps the number of *anchor
   sub_ints* per slot to 50, but since beam search converges on the
   same top-N actions early, all 40 slots' first-50 anchors are
   typically identical → all 40 aux copies hold near-identical content.
   Each anchor sub_int generates ~230 indirect raws, so 50 anchors ×
   ~230 = 11,548 iraws. The name `iraws_keep_first` is misleading — it
   keeps 50 *anchors*, not 50 iraws.

2. **`sub_accum` is dead data.** `apply_substitution_v5` reads it only
   to dedupe (sum coefficients of same sub-sector integral). The
   accumulated dict is NEVER consumed by the reduction. The final
   answer is built via `replay_full_expr` which rebuilds subs from
   scratch using `path`, ignoring `sub_accum` entirely. ~1.5 MB per
   slot of pure waste.

3. **`tabu_dict` is uncapped.** Grows ~2,000 (target, op, delta) entries
   per step. By step 5,000+ it's hundreds of MB on its own. No
   eviction.

   Plus a minor: **`path` contents for non-winning slots are dead.**
   Only `len(s.path)` is read during search; only `best_state.path` is
   replayed. The other 39 slots' tuple lists are pure waste.

**Impact:** Worker RSS grows roughly linearly with step count and
clusters at characteristic memory plateaus, regardless of integral.
Single-digit-GB at step <500, 8-12 GB at step 1000-1500, 17-20 GB at
step 2000+. Workers OOM (crash via PyTorch RuntimeError or silent
SIGKILL), drop into the orphaned-pending pile (Issue #9), and their
integrals stick in expr permanently.

**Fix paths (ranked by impact):**

1. **Hoist `aux_flat` to the beam level.** All 40 slots' aux is
   near-identical after iraws_keep_first prune; hold ONE canonical aux
   at the beam level and re-derive per-slot deltas only when slot
   paths diverge in the first-50-anchors window. Saves ~565 MB pickled,
   multiples more in live memory due to Python dict/int boxing
   overhead.
2. **Drop `sub_accum` from State_v5.** Verify `replay_full_expr`
   doesn't need it (current code only uses `path`). Removing the field
   saves the per-step copy cost and ~60 MB across the beam, more in
   memory.
3. **Drop full `path` from non-winning slots.** Carry only
   `path_len: int` per slot, promote to full tuple list only when a
   slot is set as the new `best_state` (line 900).
4. **Cap `tabu_dict`.** LRU-evict (expr_fp, action) pairs beyond N
   steps, or evict expr_fp buckets whose last access is M steps old.
5. **Memoize cu/ubm by `(raw_id, frozenset(resolved_subs_keys_used))`.**
   Less invasive than #1 but the frozenset hash at 1500+ keys is
   expensive — likely net loss.

**Verification:** Run the canonical long-runner integral
`I[1,1,1,0,0,1,3,1,-2,-1,0]` (peak 7.1 GB pre-fix at 987 steps) and
measure peak RSS after each fix. Expected reductions: hoist aux
3-5 GB, drop sub_accum 0.1 GB, drop non-winner path 0.1 GB, cap
tabu 1-2 GB.

**Detailed analysis:** See memory file
`sailir_v6_memory_bloat.md` for per-slot breakdown,
checkpoint sizes, and the empirical pattern that all 40 slots have
identical aux content modulo the slight slot-specific path divergence
in the first-50 window.

---

## How to track new issues

Append a new `## N. <one-line title>` section above with the
where/symptom/cause/impact/fix template. Don't renumber when items get
fixed — leave them in place with a "✅ FIXED <date>" line so the file
serves as a partial changelog.
