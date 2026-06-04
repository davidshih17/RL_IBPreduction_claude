# v5 iraws Design: Baseline-Order with Sparse cached

## Motivation

v4-rescue's iraws (`compute_indirect_substituted_exprkeyed_delta`) keeps only
"useful K" anchored entries to save memory. But this gives a DIFFERENT iraws
iteration order than baseline (`compute_indirect_substituted_incremental`).

Different iraws order → different `valid` action list order → with
`max_actions` truncation, the model sees different first-N subsets → softmax
denominator diverges → action_prob diverges → score drift → beam swap (we
saw this happen at step 167).

Even with `max_actions = 4000` (no truncation for baseline since baseline's
valid lists all fit), v4-rescue still gets different order if iraws structure
differs.

## Diagnostic Summary

- Baseline's iraws is already in canonical order: `(sub_int_index_in_RS,
  op_id, shift_index_in_topology)`. Verified at step 166.
- v4-rescue's iraws is a SUBSET of baseline's entries. Sorting it by canonical
  key does NOT recover baseline's order because some "first-emitter" entries
  baseline has are missing in v4 — actions then appear at later positions in
  v4's `valid` via different anchor entries.
- Action SETS are identical between v4 and baseline (verified 160/160 tasks
  at step 166).
- Model is fully permutation-equivariant (verified: same SET, different
  order → sorted logits BIT-identical).

## v5 Proposal

Drop the "drop iraws entries with no useful K" optimization. Instead:

1. **iraws_meta = baseline structure**: keep ALL entries baseline would have
   (sub_int_1's raws + sub_int_2's raws + ... + sub_int_N's raws, each
   sub_int's raws in topology shift order). NO dropping.
2. **cu cached SPARSE**: for each iraws entry, populate cached dict ONLY if
   its raw contains at least one "useful K" (K ∈ RS with sol_K ∩ expr_nm ≠ ∅).
   Otherwise, store empty `{}` for cached.

## Cost Analysis

### Memory
- iraws_meta per entry: ~60 bytes. At ~36,000 entries × 40 survivors ≈ 87 MB.
  Negligible.
- cu cached dicts: ~3 KB per populated entry. Only ~5,000 populated per
  survivor → ~15 MB per survivor × 40 ≈ 600 MB. Same as current v4-rescue.
- Empty `{}` cached entries: <100 bytes each. Negligible.
- **Total memory ≈ current v4-rescue** (NOT baseline's 4+ GB).

### Compute
- Phase A: iterates 36,000 cu entries to substitute new_sub_int. For empty
  cached, `if sub_int not in cached: return cached` is O(1) — instant skip.
  Real substitution only happens on ~5,000 populated entries. Total work ≈
  current v4-rescue.
- Phase B: enumerate new_sub_int's raws (topology iteration). Same as
  baseline.
- Per step we also need to identify which entries became useful/un-useful
  (sweep entries, check `any K in raw is in new_useful_set`). O(|iraws| ×
  avg|raw|) ≈ 36k × 15 = 540k ops per step. Cheap.
- Phase 1b iteration in enumerate_valid_actions: iterates all 36k entries
  but SC2 check (`target in cached`) returns False instantly for empty cached.
  Fast skip. Total cost dominated by ~5,000 non-empty SC2 checks.

### Correctness
- iraws iteration order = baseline's canonical order → valid lists bit-
  identical between v4-rescue-v5 and baseline.
- Phase 1b emission: entries with empty cached cannot pass SC2 → no action
  emitted from those entries. Same behavior as if entry were absent.
  ALSO same behavior as baseline's emission would be (since baseline's
  Phase 1b check is `target in cached` and cached at non-useful-K entries
  also doesn't contain target).

## Implementation Sketch

Add a new function `compute_indirect_substituted_v5` in `ibp_env.py`:
- Structurally identical to `compute_indirect_substituted_incremental` for
  building iraws_meta.
- After each step's Phase A + Phase B, sweep cu and clear (set to `{}`) the
  cached for entries whose raw has no useful K.

Or equivalently: a post-processing pass after each call to
`compute_indirect_substituted_incremental`.

Wrap the original function:
```python
def compute_indirect_substituted_v5(prev_aux, new_sub_int, new_resolved_sol,
                                    new_resolved_subs, ibp_t, li_t, shifts,
                                    raw_eq_cache, target_sector=None,
                                    expr_nm=None):
    result, aux = compute_indirect_substituted_incremental(...)  # baseline build
    cu, ubm, rid, iraws = aux
    useful_K = set(K for K, sol_K in new_resolved_subs.items()
                   if any(t in expr_nm for t in sol_K))
    new_cu = []
    for entry, c, u in zip(iraws, cu, ubm):
        sub_int, op, shift, raw = entry
        has_useful_K = any(k in useful_K for k in raw)
        if has_useful_K:
            new_cu.append(c)
        else:
            new_cu.append({})  # sparse
    return result, (new_cu, ubm, rid, iraws)
```

(Caveat: actually need to clear ubm too for consistency. And `result` is
derived from cu — needs reconstruction with cleared cached.)

## Testing

Once implemented:
1. Use the diff_thick_ckpts approach: at any step, recompute v5's iraws +
   valid, compare to baseline's valid by SET, ORDER, position-by-position.
2. Should be 100% order-equal (vs current 0% order-equal).

## Status

Pending implementation. Captured here so it doesn't get lost while we
continue investigating max_actions tradeoffs.

— captured 2026-06-03, after seeing canonical sort alone insufficient
  (test_canonical_iraws_sort.py shows v4-sorted vs baseline differ at
  position 362 of 3886).

---

## Follow-up: drop `subs` from State entirely

Captured 2026-06-03 while debugging the step-167 beam swap.

Observation: in the production hot path, NOTHING reads `subs` *values*
except two fallback functions that are no longer called once
`resolved_subs` (RS) is maintained:
- `apply_all_substitutions(raw, subs)` — replaced by `apply_resolved_subs(raw, RS)`
- `resolve_subs(subs)` — bootstrap-only; replaced by incremental `add_sub_to_resolved`

Everything else iterates `for sub_int in subs` (KEYS only):
- iraws enumeration / Phase 1b indirect raw collection (ibp_env.py:930, 992,
  1064, 1493, 2185)
- filter_subs_to_exact_sector (only the keys are needed for sector filtering)

Model input fields `sub_keys`, `sub_repl_ints`, `sub_repl_coeffs` are fed
from `subs`, but we previously verified the classifier is INSENSITIVE to
`sub_repl_ints`/`sub_repl_coeffs` (feeding random values gives identical
output). So the model effectively only uses `sub_keys`.

Dedup already keys on `resolved_subs` (beam_search_full.py:994-1000), so
no impact there.

Final expr reconstruction (`replay_path_to_full_expr`) replays raw IBP
templates from `path`, not from `subs`. No impact.

### Proposed simplification

Drop `subs` from the `State` namedtuple. Replace any references with:
- For substitution math: use `resolved_subs` (mathematically equivalent —
  keys identical, values fully resolved).
- For model `sub_keys`: use `resolved_subs.keys()` (same set).
- For model `sub_repl_ints/coeffs`: feed zeros (model verified
  insensitive) or use `resolved_subs` values (also fine).

### Memory impact

In late steps `subs` contains ~|reductions| entries with non-trivial
value dicts. Per-survivor savings depend on whether `subs` values are
materially smaller than `resolved_subs` values; at minimum we save the
outer dict + key duplication. Concrete bytes TBD; suspect this is a
secondary win vs the iraws restructuring.

### Validation plan

1. Patch `prepare_batched_input_v5` to feed `sub_repl_ints/coeffs = zeros`.
   Re-run (8,4) probe end-to-end. Expect bit-identical beam to current
   baseline. (Cheap: ~2.5h, single Condor job.)
2. If (1) passes, drop `subs` from State, replace any value reads with
   RS values, rerun. Expect bit-identical.

### Caveats

- Need to verify nothing in `delta_*` code path or `aux_flat` machinery
  silently reads `subs` values.
- Training data still has the un-resolved sol fields — irrelevant to
  inference, but if we ever retrain we'd want to decide whether to drop
  these fields from the training input too.
- Doesn't fix the v4-vs-baseline iraws-order divergence on its own; this
  is independent cleanup that can land alongside v5 or separately.

