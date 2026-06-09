"""
Count independent sectors and corner integrals in pentagon-box (TA)
after applying Kira's symmetries.

For each non-trivial sector S (from sectormappings/TA/nonTrivialSector):
  * Compute the corner integral: a_i = 1 at every bit-set position of S,
    0 elsewhere. (TA has 11 positions; sectors are 11-bit.)
  * Canonicalize it under the SymmetryGroup built from sectorSymmetries
    + sectorRelations.
  * Report the number of distinct canonical-corner forms, both globally
    and split by propagator count t.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from sailir.symmetries import (
    SymmetryGroup,
    parse_symmetries,
    sector_of,
)

TA_DIR = Path(
    "/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/examples"
    "/2-loop-pentagonbox/sectormappings/TA"
)

N_INDICES = 11
N_LOOPS = 2


def sector_corner(sector_id: int, n_indices: int = N_INDICES):
    """Corner integral: index 1 at every bit-set position of sector_id."""
    return tuple(1 if (sector_id >> i) & 1 else 0 for i in range(n_indices))


def read_non_trivial_sectors(path: Path):
    """Yield (sector_id, t) from Kira's nonTrivialSector file."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sec, t = line.split()
            out.append((int(sec), int(t)))
    return out


def main():
    sym_recs = parse_symmetries(str(TA_DIR / "sectorSymmetries"),
                                n_indices=N_INDICES, n_loops=N_LOOPS)
    rel_recs = parse_symmetries(str(TA_DIR / "sectorRelations"),
                                n_indices=N_INDICES, n_loops=N_LOOPS)
    G = SymmetryGroup.from_records(sym_recs, rel_recs, include_dots=True)
    print(f"loaded {len(sym_recs)} sectorSymmetries + {len(rel_recs)} sectorRelations")

    sectors = read_non_trivial_sectors(TA_DIR / "nonTrivialSector")
    print(f"non-trivial sectors: {len(sectors)}")

    # For each sector, build corner and canonicalize.
    sec_canon_corner = {}              # sector_id -> canonical corner tuple
    corner_to_sectors = defaultdict(list)  # canonical corner -> [sector_ids that map here]
    sec_t = {}                          # sector_id -> t

    for sec, t in sectors:
        corner = sector_corner(sec)
        # sanity: sector_of(corner) should equal sec
        if sector_of(corner) != sec:
            raise RuntimeError(f"sector_of({corner}) = {sector_of(corner)} != {sec}")
        canon = G.canonicalize(corner)
        sec_canon_corner[sec] = canon
        sec_t[sec] = t
        corner_to_sectors[canon].append(sec)

    n_distinct = len(corner_to_sectors)
    print(f"\ndistinct canonical corners across all non-trivial sectors: {n_distinct}")
    print(f"reduction factor: {len(sectors)/n_distinct:.2f}x")

    # Break down by propagator count t.
    by_t_raw = defaultdict(int)
    by_t_canon = defaultdict(set)
    for sec, t in sectors:
        by_t_raw[t] += 1
        by_t_canon[t].add(sec_canon_corner[sec])
    print("\nbreakdown by t (number of propagators):")
    print(f"  {'t':>3}  {'raw sectors':>11}  {'distinct canon corners':>23}  {'ratio':>6}")
    for t in sorted(by_t_raw):
        raw = by_t_raw[t]
        canon = len(by_t_canon[t])
        ratio = raw / canon if canon else float('nan')
        print(f"  {t:>3}  {raw:>11}  {canon:>23}  {ratio:>6.2f}")

    # Show a few example orbit collapses
    multi = [(canon, secs) for canon, secs in corner_to_sectors.items() if len(secs) > 1]
    multi.sort(key=lambda kv: -len(kv[1]))
    print(f"\ntotal corner-orbit collapses: {len(multi)} canonical reps have >1 sector")
    print(f"top 10 collapses (by orbit size):")
    for canon, secs in multi[:10]:
        secs_show = secs[:8] + (['...'] if len(secs) > 8 else [])
        print(f"  canon={canon}   sectors({len(secs)}): {secs_show}")


if __name__ == "__main__":
    main()
