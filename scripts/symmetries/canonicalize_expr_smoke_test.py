"""
Unit test for sailir.symmetries.canonicalize_expr.

Checks:
  1. Empty expression -> empty.
  2. Singleton non-orbit-merging term -> same key (canon), same coeff.
  3. Two terms in the same orbit, same coeff -> merged, sum mod prime.
  4. Two terms in the same orbit, opposite coeffs -> cancels (dropped).
  5. Three-orbit example: a real cross-sector identification
     (1,0,1,0,0,1,0) ~ (0,1,1,1,0,0,0) ; coefficients add at the rep.
  6. Apply a real trianglebox IBP to a seed, then canonicalize_expr; check
     that the term count is <= raw-application term count (symmetry merge
     should never INCREASE term count) and that no key has zero coeff.
  7. canonicalize is sign-free at the integral level: feed expr with c=1
     for each of the 6 records that map sec 37 -> sec 14 (all to the same
     target), and verify the merged coefficient is 6 (not 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from sailir.symmetries import (
    SymmetryGroup,
    apply_record,
    canonicalize_expr,
    parse_symmetries,
    sector_of,
)
from sailir.ibp_env import (
    KINEMATICS,
    PRIME,
    get_raw_equation,
    parse_templates,
    IBP_PATH,
    LI_PATH,
)

TRIANGLEBOX_DIR = Path(
    "/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples"
    "/2-loop-trianglebox/sectormappings/trianglebox"
)

TEST_PRIME = 1009  # small prime, same as SAILIR's preferred training prime


def main():
    tb_sym = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorSymmetries"),
                              n_indices=7, n_loops=2)
    tb_rel = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorRelations"),
                              n_indices=7, n_loops=2)
    G = SymmetryGroup.from_records(tb_sym, tb_rel, include_dots=True)

    p = TEST_PRIME

    # ---------- 1: empty
    print("[1] empty expr")
    out = canonicalize_expr({}, G, p)
    assert out == {}, out
    print(f"    OK -> {out}")

    # ---------- 2: singleton, not in any non-trivial orbit
    print("\n[2] singleton non-orbit-merging term")
    seed = ((1, 1, 1, 1, 1, 0, 0), 7)
    expr = {seed[0]: seed[1]}
    out = canonicalize_expr(expr, G, p)
    # (1,1,1,1,1,0,0) is in PAPER_MASTERS; we expect canon to be itself
    assert out == {seed[0]: seed[1]}, out
    print(f"    OK -> {out}")

    # ---------- 3: two terms in same orbit, same coeff
    print("\n[3] two terms in same orbit, same coeff")
    # (1,0,1,0,0,1,0) and (0,1,1,1,0,0,0) are in the same orbit
    expr = {
        (1, 0, 1, 0, 0, 1, 0): 3,
        (0, 1, 1, 1, 0, 0, 0): 4,
    }
    out = canonicalize_expr(expr, G, p)
    assert out == {(0, 1, 1, 1, 0, 0, 0): 7 % p}, out
    print(f"    OK -> {out}")

    # ---------- 4: two terms in same orbit, opposite coeffs => cancel
    print("\n[4] cancellation")
    expr = {
        (1, 0, 1, 0, 0, 1, 0): 5,
        (0, 1, 1, 1, 0, 0, 0): (p - 5),
    }
    out = canonicalize_expr(expr, G, p)
    assert out == {}, out
    print(f"    OK -> {out}")

    # ---------- 5: cross-sector identification, multi-term
    print("\n[5] cross-sector + non-orbit mixed")
    expr = {
        (1, 0, 1, 0, 0, 1, 0): 100,           # sec 37 -> canon (0,1,1,1,0,0,0)
        (0, 1, 1, 1, 0, 0, 0): 23,            # sec 14, same canon
        (1, 1, 1, 1, 1, 0, 0): 11,            # different orbit, canon=self
    }
    out = canonicalize_expr(expr, G, p)
    expected = {
        (0, 1, 1, 1, 0, 0, 0): (123) % p,
        (1, 1, 1, 1, 1, 0, 0): 11,
    }
    assert out == expected, f"got {out}\nexp {expected}"
    print(f"    OK -> {out}")

    # ---------- 6: real IBP application then canonicalize
    print("\n[6] real trianglebox IBP application + canonicalize_expr")
    ibp_t = parse_templates(IBP_PATH)
    li_t = parse_templates(LI_PATH)
    # pick a seed in sector 37 (cross-sector relations exist) and apply IBP #0
    seed = (1, 0, 1, 0, 0, 1, 0)
    raw = get_raw_equation(ibp_t, li_t, ibp_op=0, seed=seed)
    print(f"    raw IBP: {len(raw)} terms")
    out = canonicalize_expr(raw, G, PRIME)
    print(f"    canon:   {len(out)} terms")
    # merge can only decrease (or preserve) term count
    assert len(out) <= len(raw), f"{len(out)} > {len(raw)}"
    for k, v in out.items():
        assert v != 0, k
    print("    OK: no zero coeffs; canon term count <= raw term count.")

    # ---------- 7: sign-free check (consistency of 6 sec37->sec14 records)
    print("\n[7] sign-free integral identity check")
    sec37_recs = [r for r in tb_rel if r.source_sector == 37]
    assert len(sec37_recs) == 6, len(sec37_recs)
    src = (1, 0, 1, 0, 0, 1, 0)
    expr = {}
    for r in sec37_recs:
        img = apply_record(src, r)
        assert img is not None
        expr[img] = (expr.get(img, 0) + 1) % p  # add 1 per record
    out = canonicalize_expr(expr, G, p)
    # all 6 records should hit the same target tuple, so coeff = 6
    assert out == {(0, 1, 1, 1, 0, 0, 0): 6}, out
    print(f"    OK -> {out}   (all 6 records consistent; total coeff = 6)")

    print("\nCANONICALIZE_EXPR SMOKE TEST: OK")


if __name__ == "__main__":
    main()
