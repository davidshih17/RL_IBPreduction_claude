"""
Smoke test for sailir.symmetries.parse_symmetries.

Loads sectorSymmetries + sectorRelations for both trianglebox (n_indices=7)
and pentagon-box TA (n_indices=11), and prints:
  - record counts (cross-checked against the HOWTO numbers we know)
  - sign distribution
  - sym_dots distribution
  - distribution of source/target sectors
  - a few example records, including one applied to a concrete integral
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Make sure we can import sailir.* from the worktree
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent  # .../SAILIR_phase2
sys.path.insert(0, str(ROOT))

from sailir.symmetries import (
    SymmetryRecord,
    parse_symmetries,
    sector_of,
    apply_record,
)

TRIANGLEBOX_DIR = Path(
    "/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples"
    "/2-loop-trianglebox/sectormappings/trianglebox"
)
PENTAGONBOX_DIR = Path(
    "/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples"
    "/2-loop-pentagonbox/sectormappings/TA"
)


def summarize(name: str, recs):
    print(f"--- {name}: {len(recs)} records ---")
    if not recs:
        return
    signs = Counter(r.sign for r in recs)
    dots = Counter(r.sym_dots for r in recs)
    src_sectors = sorted({r.source_sector for r in recs})
    tgt_sectors = sorted({r.target_sector for r in recs})
    cross_sector = sum(1 for r in recs if r.source_sector != r.target_sector)
    print(f"  sign distribution:    {dict(signs)}")
    print(f"  sym_dots distribution:{dict(dots)}")
    print(f"  num distinct source sectors: {len(src_sectors)}")
    print(f"  num distinct target sectors: {len(tgt_sectors)}")
    print(f"  records with source != target: {cross_sector}")
    print("  first 3 records:")
    for r in recs[:3]:
        print(f"    src={r.source_sector} -> tgt={r.target_sector}"
              f"  sign={r.sign:+d}  dots={r.sym_dots}  n_props={r.n_props}"
              f"  ing={r.ing}")


def apply_demo(name: str, recs, integral):
    print(f"--- {name}: applying records to integral {integral} (sector {sector_of(integral)}) ---")
    applied = 0
    for r in recs[:20]:
        new_int = apply_record(integral, r)
        if new_int is None:
            continue
        applied += 1
        print(f"    rec(src={r.source_sector}->tgt={r.target_sector}, "
              f"det={r.sign:+d}, dots={r.sym_dots}, ing={r.ing}): "
              f"I[{integral}] = I[{new_int}]"
              f" (target sector={sector_of(new_int)})")
        if applied >= 8:
            break
    if applied == 0:
        print("    (no record in first 20 was applicable to this integral)")


def main():
    # ----- trianglebox -----
    print("=" * 70)
    print("TRIANGLEBOX (n_indices=7, n_loops=2)")
    print("=" * 70)
    tb_sym = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorSymmetries"),
                              n_indices=7, n_loops=2)
    tb_rel = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorRelations"),
                              n_indices=7, n_loops=2)
    summarize("trianglebox/sectorSymmetries", tb_sym)
    summarize("trianglebox/sectorRelations", tb_rel)
    # Known from HOWTO: 40 sectorSymmetries, 12 sectorRelations
    assert len(tb_sym) == 40, f"expected 40 sectorSymmetries, got {len(tb_sym)}"
    assert len(tb_rel) == 12, f"expected 12 sectorRelations, got {len(tb_rel)}"
    print(">> trianglebox counts match HOWTO (40 + 12)")

    # Demo: apply sectorSymmetries to an integral in sector 14 with NO ISP usage
    apply_demo("trianglebox sym", tb_sym, integral=(0, 1, 1, 1, 0, 0, 0))
    # Demo: an integral in sector 37 (bits 0,2,5 set) to hit sectorRelations
    apply_demo("trianglebox rel", tb_rel, integral=(1, 0, 1, 0, 0, 1, 0))

    # ----- pentagon-box TA -----
    print()
    print("=" * 70)
    print("PENTAGONBOX TA (n_indices=11, n_loops=2)")
    print("=" * 70)
    ta_sym = parse_symmetries(str(PENTAGONBOX_DIR / "sectorSymmetries"),
                              n_indices=11, n_loops=2)
    ta_rel = parse_symmetries(str(PENTAGONBOX_DIR / "sectorRelations"),
                              n_indices=11, n_loops=2)
    summarize("TA/sectorSymmetries", ta_sym)
    summarize("TA/sectorRelations", ta_rel)
    # Actual line counts (verified with wc -l): 111 sectorSymmetries,
    # 21 sectorRelations. (The files symmetries / relations are a
    # separate Kira internal block-based encoding of the same identities.)
    assert len(ta_sym) == 111, f"expected 111 sectorSymmetries, got {len(ta_sym)}"
    assert len(ta_rel) == 21, f"expected 21 sectorRelations, got {len(ta_rel)}"
    print(">> TA counts match file line counts (111 + 21)")

    # Demo: apply on a TA integral in sector 53 (bits 0,2,4,5 set = 4 props)
    apply_demo("TA", ta_sym,
               integral=(1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0))
    # And cross-sector demo: integrals in source sectors of sectorRelations
    src_sectors_rel = sorted({r.source_sector for r in ta_rel})
    print(f"  TA sectorRelations source sectors: {src_sectors_rel}")

    print()
    print("PARSER SMOKE TEST: OK")


if __name__ == "__main__":
    main()
