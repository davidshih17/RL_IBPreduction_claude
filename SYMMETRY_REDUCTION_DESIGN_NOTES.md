# Symmetry-Enhanced Reduction — Design Notes

Design decisions from the symmetry investigation (pentagon-box "TA" family,
`topology_input/pentagonbox_nosym`). Kira symmetry files:
`results/kira_reduce_161/sectormappings/TA/{sectorSymmetries,sectorRelations}`.

## Background facts (established & verified)

- **What symmetries are:** loop-momentum relabelings (affine in `k1,k2`) — loop
  reflections (`k1 → −k1−…`, `k2 → −k2−…`) and loop exchange (`k1 ↔ k2`), plus
  compositions. Each induces a permutation of the propagators `D1…D8`.
- **`sectorSymmetries`** (source==target sector) = within-sector automorphisms;
  **`sectorRelations`** (source≠target) = cross-sector (lateral, same prop-count).
- **Top sector 255 is ASYMMETRIC** (no records → trivial group). Symmetry lives
  ONLY in sub-sectors, per-sector, with incompatible groups. Sector 57 = {0,3,4,5}
  is the loop-exchange sector (both loops become identical bubbles) → order-8
  non-abelian → genuine 2D irreducible ISP block. It still reduces fine to lower
  weight; the 2D irrep only blocks "collapse to a single same-weight representative".
- **What a symmetry does to a numerator** — exactly 4 things (it acts linearly on
  scalar products): (1) ISP→same slot, (2) ISP→other ISP slot(s) [same weight],
  (3) hits a present propagator → cancels it → lower sector, (4) → constant → lower
  numerator degree. It **never raises** numerator degree or propagator count; a
  numerator can only MOVE to a new slot (degree conserved), never be created from a
  pure-denominator integral.
- **`P_S = (1/|G|) Σ_g M_g`** (same-`(r,s)`-weight part of the symmetry action) is
  the projector onto the fully-symmetric (invariant) subspace at each (sector,
  weight-level). Identity: `I[a] = P_S·I[a] + (strictly-lower total-weight)`.
  The invariant-subspace dim = "survivors"; everything else reduces to lower weight.
- **CRITICAL:** symmetry ALONE does NOT reduce to masters — IBP is required. The
  symmetry descent `I[a] = Σ P_S·I[b]` terminates at the CORNER integrals (pure
  denominators, no symmetry tail); IBP takes over there. "symmetry → corners; IBP → masters."

## Design 1 — RECOMMENDED: symmetry as a free routing layer under the workers

Keep individual integrals everywhere (no reformulation). `P_S` is **linear**, so a
symmetry relation is just an ordinary linear relation the substitution engine already
consumes — **no engine change**.

1. **Precompute** per (sector, weight-level): the **survivors** (symmetry-irreducible
   individual integrals) + for every non-survivor a clean **strictly-lowering rewrite
   rule** `I → (lower individual integrals)`. This is `P_S` in the individual-integral
   basis (see `reduction/individual_rules.py`).
2. **Add rules to `resolved_subs`** (they're linear relations; nothing new).
3. **Route dispatch:** symmetry-reducible integral → apply rule (free, no worker);
   survivor → dispatch a model-guided IBP worker.

- Realizes the `P_S` chain implicitly; bottoms out at corners.
- Handles "different `P_S` per sector" natively = a per-sector rule-table lookup.
- **Avoids the live-lock** we hit before (precomputed clean strictly-lowering rules,
  not raw same-weight relations applied ad hoc).
- **Win:** reduce ONE orbit representative, not all orbit members → saves duplicate
  worker subtrees. Savings concentrated in symmetric SUB-sectors (top 255 gets none).

## Design 2 — global block-diagonalized (symmetry-adapted) IBP — NOT applicable here

Project the whole IBP system with `P_S`: antisymmetric relations → `0=0` (vanish),
giving a strictly smaller system (fewer objects AND fewer relations).

- **Does NOT apply to the pentagon-box:** there is NO global `P_S` (top sector 255 is
  asymmetric; per-sub-sector groups are incompatible; IBP relations cross sub-sector
  boundaries). Only makes sense for families whose TOP sector is itself symmetric.
- Even where it applies it wants a symmetric-combination basis. Linearity means you
  *could* stay in individual integrals — but with no global symmetry there's nothing
  to globally project, so it degenerates into Design 1 anyway.

## Bottom line

Because `P_S` is linear and the global symmetry is trivial, the two designs collapse:
do **Design 1** — precompute per-sector clean symmetry relations, drop them in
`resolved_subs`, route dispatch to reduce one representative per orbit. Zero change to
the substitution engine. `P_S` is the *explanation*; operationally it's linear relations
in the pool, sector by sector. Measure the routed-for-free fraction on a real target
before building (top sector helps nothing; symmetric sub-sectors help a lot).

## Measured budget (concrete, `reduction/design1_budget.py`)

- **Symmetric sub-sectors: 136 of 238** (prop-count 2–6), concentrated at LOW prop-count:
  2-prop 27/28 (96%), 3-prop 35/56, 4-prop 46/70, 5-prop 23/56, 6-prop 5/28 (18%);
  7/8-prop near-top are asymmetric. ⇒ symmetry pays off DEEPER in the tree.
- **Reducible (free rule) vs survivor (worker)**, summed over symmetric sectors, corner denominators:
  - s=1: total 1008 → **732 (73%) get a rule**, 276 (27%) survivors.
  - s=2: total 4321 → **2666 (62%) get a rule**, 1655 (38%) survivors.
- **Per-rule generation: 72 µs** (~13.8k rules/sec) = image over the group + total-weight row-reduce.
- **Budget:** negligible. Full no-dot baseline (~5k rules) < 1 s; even the entire symmetric-sector
  seed set up to (r=10,s=4) with dots (~millions) is only minutes single-threaded. Lazy+cache pays
  only for visited integrals. Each freed integral avoids a full IBP worker subtree (sec–hrs).
- **Caveat:** raw per-level fractions. Real worker-savings on a target = how much of THAT target's
  reduction tree lands in the 136 symmetric sectors — replay a logged run to get the number.

## Replay against real v7 runs (`reduction/replay_fixed2.py`)

Replayed 4 real single-step v7 reductions (probe_84/74/monster/memhog, 567 distinct
integrals touched). **These UNDERSTATE the full reduction** (single-step workers stay in
their own active-bucket sector; runs only capture the top 1–2 descent levels):
- reduced (top-worker targets): **0% free** — all in high ASYMMETRIC sectors (5–7 prop).
- passengers (one level down): **5.3% free**, 23% symmetric-survivor, 71.6% asymmetric.
- all touched: **2.6% free**, 11.5% survivor, 85.9% asymmetric.
- Trend: free-routable rises 0%→5.3% descending ONE level; symmetry lives at low prop-count
  (2-prop 96% … 6-prop 18% symmetric), which shallow runs don't reach. 2.6% is a FLOOR.

**Methodology gotcha (cost me two wrong 0% results):** a sector's automorphisms FIX its
corner (`image_unsigned(corner)=={corner:1}`) — they act nontrivially only on numerators.
So "symmetric" is `len(sym_autos(corner))>1`, NOT "corner image ≠ corner". And use
`secbits()` over `range(ND)=8` for the sector tuple (an 11-length binary compares unequal to
the 8-length csec → everything falsely reads asymmetric). Sanity: `{1,2,4,7}`→4 autos,
`{0,3,4,5}`→20, `{0,1,2,3}`→5.

**For the real number:** need a FULL orchestrator run's complete dispatched-integral set
(all descent levels to the 2–4-prop master sectors), not single-step worker pkls.

## Closure test — symmetry descent vs orchestrator reduction (`reduction/final_closure3.py`)

VERIFIED: for real list_TA integrals in symmetric sectors, reduce the symmetry-descent
expression `I = P_S·I + Σ cⱼ Cⱼ` to masters and compare to I's direct orchestrator
reduction R(I). **10/10 agree on EVERY genuine (non-corner) master.** The `P_S` descent
formula is correct and consistent with the actual reduction.

- Residual difference = pure-denominator CORNER terms only. These are the "fake masters"
  of the `--no-paper-masters-only` basis (list_TA_reductions.pkl uses it): IBP keeps them
  as independent masters but symmetry relates them. So descent lands on a symmetric combo
  of corners; IBP-only R(I) picks one corner rep — equal as values, different in the
  redundant basis. This is the "symmetry gives relations beyond IBP" signature.
- The 10 uncovered corners, when reduced via fresh orchestrator jobs (results/corner_reductions/),
  ALL came back as masters in ~0 s / 0 Condor workers — confirming they're exactly the
  spurious corner-masters `--paper-masters-only` drops.
- Reduction tables used: results/meta_reduce/list_TA_reductions.pkl (996→75 masters,
  no-paper-masters-only) + replay/reduction_cache.pkl (103k one-step reductions).
- Gotcha: descent must average over APPLICABLE autos only (image_unsigned returns None
  when a symmetry is inapplicable to an integral's numerator config); divide by that count
  and add the identity if absent.

## Accurate savings measurement (`reduction/measure_savings.py`)

Classified ALL 103,626 integrals the orchestrator actually reduced (replay/reduction_cache.pkl
keys = the real deduped work) into FREE (symmetry rewrite rule, no worker) vs WORKER:
- **FREE: 24,800 (23.9%)** — symmetry saves ~24% of the reduced-integral workload.
- survivor (symmetric, worker): 43,535 (42.0%);  asymmetric (worker): 35,291 (34.1%).
- Concentrated at low prop-count: 2-prop 31%, 3-prop 33%, 4-prop 23%, 5-prop 9%, 6-prop 2%,
  7/8-prop 0%. Bulk of work is 3–4 prop (67k of 103k) → that's where the 24% comes from.

Caveats:
1. Per-integral (deduped) count, NOT wall-clock. Freed integrals skew CHEAP (low-sector), so
   wall-clock saving is likely < 24% (keep the expensive high-sector workers); but subtree
   pruning could push it higher. True wall-clock needs per-worker timing.
2. 42% "survivors" is inflated by CURRENT symmetry data: high-degree-numerator integrals hit
   INAPPLICABLE placeholder records (image_unsigned=None). A full momentum-map symmetrization
   would convert some survivors→free, so 24% is a floor for what symmetry could do and the
   realized number for the current _transforms implementation.

## Applicability-gap investigation (`reduction/placeholder_recon.py`, `quantify_gap.py`)

Attempted to close the "applicability gap" (image_unsigned=None on high-degree numerators)
by reconstructing each symmetry's loop relabeling from its propagator permutation, then
deriving the full momentum map. OUTCOME: **not closable this way, and the gap is small.**

- Reconstruction WORKS for all 99 momentum-map records, FAILS for all 33 placeholders —
  because placeholders are "dots symmetries" (sym_dots=1) that are NOT loop relabelings.
  Concrete proof: sector-55 placeholder swaps D1↔D3 while FIXING D2; no relabeling
  k→Mk+shift can do that. So there is no momentum map to derive.
- Scope of the gap (of 103,626 reduced integrals): **82.1% full-M-capable** (momentum-map
  record → already handles ANY numerator degree), **6.3% placeholder-only** (the gap),
  11.6% no-symmetry. So the gap is only 6.3%, NOT the driver of the 42% survivors.
- CORRECTION to earlier claim: the 42% survivors are NOT "inflated by the applicability
  gap" — in the 82% full-M region the survivors are GENUINE orbit reps. And "ceiling
  62–73%" was a per-level CORNER figure; over the full workload (with dots/high-degree
  genuine survivors), **24% is the realized-and-near-ceiling saving** for this symmetry data.
- Closing the 6.3% would need Kira to re-emit the dots symmetries' full numerator action
  (they're value identities but not propagator relabelings) — bigger than a canonicalize
  patch, for ≤6.3% gain. Not worth it.

## CORRECTION — the gap IS closable (`reduction/recon_from_swaps.py`, `corrected_savings.py`)

Earlier "not closable / drop the placeholders" was WRONG. The placeholders' true relabelings
DO exist; the fix is to reconstruct them from the placeholder's clean SWAPS (props that map to
a DIFFERENT prop), NOT the literal `ing` (whose 'fixed' entries can be fake — a prop that really
maps to a combination). Then `derive_transform` gives the full momentum map.

- `recon_from_swaps.reconstruct(ing, present_props)`: 99/99 momentum-map self-check; **26/33
  placeholders reconstructed** (7 have no swaps → dropped safely). Sector-55 recovers exactly
  `k1→−(k1+p1+p2)`, giving `D2 → D1−D2+D3` (a combination), matching the hand-derivation.
- This is ALSO a CORRECTNESS FIX. The old lossy import (partial `M`, fake `D2→D2`) produces
  WRONG reductions: `image_unsigned(I[2,1,1,0,1,1])` → `I[1,1,2,…]` (false — verified they reduce
  to different-prop-count masters). The true `M` returns **None** for D2-present (correct reject)
  and applies only where D2 is absent (`I[2,0,1,0,1,1] → I[1,0,2,0,1,1]`, correct).
- **Corrected free-routing over the 103k cache: 21.5%, all CORRECT**
  — vs naive 24% (which included ~2.5% of WRONG placeholder reductions) and 19.7% (full-M only).
  So reconstruction adds ~+1.8% correct coverage AND removes the ~2.5% wrong reductions.
- **Action:** in `canonicalize._build_engine_src`, for placeholder records call
  `recon_from_swaps.reconstruct` → `derive_transform` instead of `ing_to_Mc`; drop records with no
  swaps. This both fixes latent wrong reductions and lifts correct savings 19.7%→21.5%.
