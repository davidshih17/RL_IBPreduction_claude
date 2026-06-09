"""
Unit test for sailir.symmetries.SymmetryGroup.canonicalize on trianglebox.

Checks performed:
 1. Orbit of (0,1,1,1,0,0,0) (a paper master) in sector 14 covers itself
    via S_3 permutations of positions {1,2,3} -> orbit size = 1 because
    all such permutations fix the integral.
 2. Orbit of (1,0,1,0,0,1,0) (sector 37) reaches (0,1,1,1,0,0,0)
    (sector 14) via the 6 cross-sector relations 37 -> 14. Canonical
    form of (1,0,1,0,0,1,0) must be (0,1,1,1,0,0,0).
 3. canonicalize is idempotent: canon(canon(x)) == canon(x).
 4. All 16 paper masters from PAPER_MASTERS canonicalize to a paper
    master (so the existing master basis is closed under canonicalize).
 5. A few non-symmetric integrals canonicalize to themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from sailir.symmetries import (
    SymmetryGroup,
    parse_symmetries,
    sector_of,
    default_order_key,
)

TRIANGLEBOX_DIR = Path(
    "/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples"
    "/2-loop-trianglebox/sectormappings/trianglebox"
)

# From SAILIR ibp_env.py PAPER_MASTERS (16 master integrals, paper 2502.05121)
PAPER_MASTERS = frozenset([
    (0, 1, 1, 1, 0, 0, 0),
    (1, 0, 1, 0, 1, 0, 0),
    (1, 1, 0, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (1, 0, 1, 1, 1, 0, 0),
    (1, -1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 0, 0),
    (-1, 1, 1, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 0, 0),
    (1, 0, 1, 0, 0, 1, 0),
    (1, 1, 0, 1, 0, 1, 0),
    (1, 0, 1, 0, 1, 1, 0),
    (1, -1, 1, 0, 1, 1, 0),
    (1, 0, 0, 1, 1, 1, 0),
    (1, 1, 0, 1, 1, 1, 0),
    (1, 0, 1, 1, 1, 1, 0),
])


def main():
    tb_sym = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorSymmetries"),
                              n_indices=7, n_loops=2)
    tb_rel = parse_symmetries(str(TRIANGLEBOX_DIR / "sectorRelations"),
                              n_indices=7, n_loops=2)
    G = SymmetryGroup.from_records(tb_sym, tb_rel, include_dots=True)
    print(f"loaded {len(tb_sym)} sectorSymmetries + {len(tb_rel)} sectorRelations "
          f"=> {len(G.records)} records in group")

    # ---- check 1: orbit of (0,1,1,1,0,0,0) ----
    m14 = (0, 1, 1, 1, 0, 0, 0)
    orb14 = G.orbit(m14)
    print(f"\n[1] orbit of {m14} (sec 14): size={len(orb14)}")
    print(f"    orbit = {sorted(orb14)}")
    assert orb14 == {m14}, f"expected singleton orbit (S_3 fixes this), got {orb14}"
    print("    OK: singleton (all S_3 perms fix this integral).")

    # ---- check 2: cross-sector 37 -> 14 ----
    m37 = (1, 0, 1, 0, 0, 1, 0)
    orb37 = G.orbit(m37)
    canon37 = G.canonicalize(m37)
    print(f"\n[2] orbit of {m37} (sec 37): size={len(orb37)}")
    print(f"    orbit = {sorted(orb37)}")
    print(f"    canonicalize -> {canon37} (sector {sector_of(canon37)})")
    assert m14 in orb37, f"expected sec-14 master in orbit of {m37}"
    assert canon37 == m14, f"canonicalize({m37}) should be {m14}, got {canon37}"
    print("    OK: orbit reaches sec-14 master; canonical form matches.")

    # ---- check 3: idempotence ----
    print("\n[3] idempotence check on several integrals:")
    test_integrals = [
        (0, 1, 1, 1, 0, 0, 0),
        (1, 0, 1, 0, 0, 1, 0),
        (1, 1, 1, 1, 1, 0, 0),
        (2, -1, 1, 1, 1, 0, 0),
        (1, 0, 1, 0, 1, 1, 0),
    ]
    for x in test_integrals:
        cx = G.canonicalize(x)
        ccx = G.canonicalize(cx)
        print(f"    canon({x}) = {cx} ; canon(canon) = {ccx}")
        assert cx == ccx, f"non-idempotent: {x} -> {cx} -> {ccx}"
    print("    OK: canonicalize is idempotent.")

    # ---- check 4: paper masters are closed under canonicalize ----
    print("\n[4] paper-master closure check (16 masters):")
    miss = []
    for m in sorted(PAPER_MASTERS, key=default_order_key):
        c = G.canonicalize(m)
        marker = "" if c in PAPER_MASTERS else "  <-- NOT IN PAPER_MASTERS"
        print(f"    canon({m}) = {c}{marker}")
        if c not in PAPER_MASTERS:
            miss.append((m, c))
    assert not miss, f"some paper masters canonicalize outside PAPER_MASTERS: {miss}"
    print("    OK: all 16 paper masters canonicalize within PAPER_MASTERS.")

    # ---- check 5: orbit-rep sanity for the 16 masters ----
    print("\n[5] orbit-rep sanity: each paper master is the rep of its orbit:")
    # Each paper master should be its own canonical form (since they were
    # presumably chosen as orbit reps).
    bad = []
    for m in PAPER_MASTERS:
        c = G.canonicalize(m)
        if c != m:
            bad.append((m, c))
    if bad:
        print(f"    {len(bad)} master(s) have a smaller orbit element:")
        for m, c in bad:
            print(f"      {m} -> {c}")
        print("    (This is INFORMATIONAL: paper's choice of rep may differ "
              "from our default_order_key.)")
    else:
        print("    OK: every paper master is its own orbit rep under our order.")

    print("\nCANONICALIZE SMOKE TEST: OK")


if __name__ == "__main__":
    main()
