# THE TOTAL ORDER — DESIGN DECISION (2026-07-10)

## The order is: SECTOR RANK first, then (r,s), then lex-weight |a|.

This is the single authoritative statement of the integral ordering for the SAILIR
reduction pipeline. Decided 2026-07-10 (dshih): **sector rank takes precedence
everywhere** — in the workers' success/target logic, in the symmetry router, in the
training-data target selection, and in every comparison of "which integral is
lower/reduced-toward".

Comparison semantics (an integral A is ELIMINATED BEFORE / sits ABOVE B when):

1. **sector_rank(sector(A)) > sector_rank(sector(B))**   — sector rank, senior key
2. tie → **(r,s)(A) > (r,s)(B)**                          — coarse weight (r = sum of
   denominator powers, s = sum of |numerator powers|), compared lexicographically
3. tie → **|a|(A) vs |a|(B)** lexicographically           — the existing abs-tuple
   tiebreak (load-bearing for confluence; do NOT drop it)

The SURVIVOR of any set (the member everything else reduces toward) is the MINIMUM
in this order.

## sector_rank: the contract

`sector_rank(S)` is any fixed total order on the 255 sector masks satisfying:

  (i)  **subsector ⇒ strictly lower rank.** If S' ⊂ S (proper subset of propagators)
       then rank(S') < rank(S). Guaranteed by making the propagator COUNT t(S) the
       senior component. This is what makes IBP steps descend: IBP never leaves the
       sector cone, and every cross-sector IBP term is in a proper subsector.
  (ii) **canonical orbit representative is rank-minimal within its clean orbit.**
       For every clean corner orbit (results/canonical_sectors_tkey.pkl), the rep
       (= the survivor corner, see canonical_rep.py) ranks strictly below every
       non-rep member. This is what makes sector-changing symmetry rewrites descend,
       and gives the HARD guarantee that only canonical sectors are ever dispatched.
  (iii) total (any tie-break completing it, stable across runs).

Reference implementation (lex tuple, smaller = lower rank):

    rank(S) = ( t(S),                 # number of propagators
                rep_of[S],            # groups orbit members together
                S != rep_of[S],       # rep first within its orbit
                S )                   # stable total tie-break

with `rep_of` from `results/canonical_sectors_tkey.pkl` (built by
`build_canonical_sectors_tkey.py`, gated by `verify_canonical_rep.py`).

## Why (measured)

The legacy production order `κ = (-r, -s, |a|)` has NO sector component. On dotted /
numerator integrals its |a| tiebreak compares powers in permuted slots, so the
κ-survivor of a symmetry orbit occasionally sits in a NON-canonical sector and gets
dispatched there. Measured leakage of dispatched worker targets into non-canonical
sectors (post-strip A/B runs, 2026-07-07): m1 12/455 (2.6%), m2 6/137 (4.4%),
m3 33/222 (15%) — vs baselines 2%/57%/90%. Sector-senior ordering eliminates this
leakage by construction: within a clean orbit at any dot/numerator pattern, the
canonical-sector image always outranks (sits below) the non-canonical original,
because sector rank is compared before anything else.

## Canonical masters (part of the same package)

Kira's paper masters are labeled in KIRA's preferred sectors; ours is a different,
self-contained convention, and exactly one pentagonbox master (sector 152) sits in a
sector our order does NOT pick as canonical (161). Under SAILIR_SECTOR_RANK=1 the
master basis is therefore the SYMMETRY-IMAGE of the paper basis in OUR canonical
sectors (`reduction/canonical_masters.py`, applied by both orchestrator and worker
at startup; exact 1-term coefficient-1 identities, dictionary back to Kira applied
to the final expression at output time only). Without this, the merged corner of a
mismatched orbit is unreducible and its worker hangs forever — the m1/m3 stuck-corner
pathology (I[1,0,0,0,0,1,0,1,0,0,0], diagnosed 2026-07-11).

## Deployment status — read this before touching any order code

The decision is adopted; the MIGRATION IS NOT DONE. As of 2026-07-10 the production
code still runs the legacy κ:

  - reduction/beam_search_v7.py::_target_key      (worker success/strip logic)
  - reduction/symmetry_route.py::tkey             (router pivoting)
  - reduction/canonical_rep.py::tkey              (orbit survivor)
  - training-data target selection (data_gen)

Because router and workers MUST share one order (the confluence requirement — see
notes/symmetry_inference_routing.tex, "Confluence" section: mixed orders deadlocked a
real reduction), the switch to sector-senior ordering must land JOINTLY in all of the
above plus the training data, i.e. with the symmetry-enhanced data-gen + retrain.
NEVER switch one component alone.
