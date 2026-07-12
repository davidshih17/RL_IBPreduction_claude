#!/usr/bin/env python
"""CANONICAL MASTERS — the master basis of the canonical-sector pipeline.

Kira's paper masters are defined in KIRA's preferred sectors; our pipeline
canonicalizes sectors by OUR order (ORDERING.md). Where the two conventions
disagree, the orbit's merged corner is not a paper master and no worker can reduce
it (IBP cannot leave the sector cone; the identity to the paper master ascends the
order) — the m1/m3 hang, stuck on I[1,0,0,0,0,1,0,1,0,0,0] (sector 161, whose
paper master lives in sector 152).

DECISION (dshih, 2026-07-11): transform Kira's paper masters INTO our canonical
sectors using the symmetries — the self-consistent choice; the ordering and
canonicalization stay self-contained and Kira's basis is reached through an exact
dictionary at output time. For pentagonbox exactly ONE master moves (sector 152 ->
161), it is a pure corner, and its image is a single integral with coefficient 1.

apply_canonical_masters() -> dict {canonical_master: kira_master} (identity entries
excluded); it swaps ibp_env's master basis in place. Call it AFTER
init_from_topology, in BOTH the orchestrator and every worker, whenever
SAILIR_SECTOR_RANK=1 (one package with the sector-senior order).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))


def build():
    """Compute (canonical_master_set, dictionary {canonical: kira}) from the current
    ibp_env master basis. Requires every moved master's image to be a single
    integral with coefficient 1 (true here: the only moved master is a corner);
    raises loudly otherwise so a future family extension is a conscious decision."""
    from sailir import ibp_env
    from canonicalize import image_unsigned
    import sector_canon_maps
    with open(os.path.join(ROOT, "results/canonical_sectors_tkey.pkl"), "rb") as f:
        canon = set(pickle.load(f)["canonical"])
    maps = sector_canon_maps.load()
    new_set = set(); dictionary = {}
    for m in ibp_env.MASTERS_SET:
        S = sum(1 << k for k in range(8) if m[k] > 0)
        if S in canon:
            new_set.add(tuple(m)); continue
        M, c = maps[S]
        img = image_unsigned(tuple(m), M, c)
        if img is None or len(img) != 1 or next(iter(img.values())) % 1009 != 1:
            raise RuntimeError(f"master {list(m)} (sector {S}) has no single-term "
                               f"coefficient-1 canonical image: {img} — extend the "
                               f"canonical-masters scheme before proceeding")
        (j, _co), = img.items()
        new_set.add(j); dictionary[j] = tuple(m)
    return frozenset(new_set), dictionary


_applied = None


def apply_canonical_masters():
    """Swap ibp_env's master basis to the canonical-sector images (idempotent).
    Returns the dictionary {canonical_master: kira_master} for output translation."""
    global _applied
    if _applied is None:
        from sailir import ibp_env
        new_set, dictionary = build()
        ibp_env.set_masters(new_set)
        _applied = dictionary
    return _applied


def translate_to_kira(expr, dictionary):
    """Rename canonical masters back to Kira's basis in a final expression (exact
    1-term identities, coefficient 1). Output-time only — never a cache entry."""
    return {dictionary.get(k, k): v for k, v in expr.items()}


if __name__ == "__main__":
    from sailir import ibp_env
    from sailir.topology import Topology
    ibp_env.init_from_topology(Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox")))
    ibp_env.set_prime(1009)
    before = set(ibp_env.MASTERS_SET)
    d = apply_canonical_masters()
    after = set(ibp_env.MASTERS_SET)
    with open(os.path.join(ROOT, "results/canonical_sectors_tkey.pkl"), "rb") as f:
        canon = set(pickle.load(f)["canonical"])
    bad = [m for m in after if sum(1 << k for k in range(8) if m[k] > 0) not in canon]
    print(f"masters before: {len(before)}, after: {len(after)}, moved: {len(d)}")
    for k, v in d.items():
        print(f"  canonical {list(k)}  <->  kira {list(v)}")
    print(f"canonical masters in non-canonical sectors: {len(bad)}")
    I0 = (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0)
    ibp_env.set_paper_masters_only(True)
    print(f"is_master(stuck corner I0) now: {ibp_env.is_master(I0)}")
    ok = (len(before) == len(after) and len(d) == 1 and not bad
          and ibp_env.is_master(I0))
    print("ALL PASS" if ok else "FAIL")
