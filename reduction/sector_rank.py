#!/usr/bin/env python
"""SECTOR RANK — the senior key of the adopted total order (see ORDERING.md).

rank tuple per sector S:  ( t(S), rep_of[S], S != rep_of[S], S )
  t(S)            propagator count  -> every proper subsector ranks below its parent
                  (IBP never leaves the sector cone, so IBP steps always descend)
  rep_of[S]       groups clean-orbit members together
  S != rep_of[S]  the canonical representative first within its orbit
                  (sector-changing symmetry rewrites always descend; only canonical
                  sectors can survive)
  S               stable total tie-break

RANK_IDX[S] = position of S's rank tuple in ascending order (0..2^N_DEN-2 over
masks 1..2^N_DEN-1; RANK_IDX[0] = -1 so the no-propagator "sector" of
pure-numerator tuples sorts below everything). Smaller index = lower rank =
reduced TOWARD.

TOPOLOGY-KEYED (SAILIR_TOPOLOGY, default pentagonbox — bit-identical to the
original hardwired module): rep_of comes from topo_config.CANON_PKL
(pentagonbox: canonical_sectors_tkey.pkl; gravity3L: canonical_sectors_GR.pkl).
This module has its own gate: run it directly to assert both rank-contract
clauses for the ACTIVE topology.
"""
import pickle
import topo_config as _tc

N_MASKS = 1 << _tc.N_DEN


def _build():
    with open(_tc.CANON_PKL, "rb") as f:
        rep_of = pickle.load(f)["rep_of"]
    masks = sorted(range(1, N_MASKS),
                   key=lambda S: (bin(S).count("1"), rep_of[S], S != rep_of[S], S))
    idx = [0] * N_MASKS
    idx[0] = -1
    for pos, S in enumerate(masks):
        idx[S] = pos
    return idx, rep_of


RANK_IDX, REP_OF = _build()


def sector_rank(mask):
    """Integer rank of a sector mask; smaller = lower = reduced toward."""
    return RANK_IDX[mask]


if __name__ == "__main__":
    # gate (i): every proper subsector ranks strictly below its parent
    bad_sub = 0
    for S in range(1, N_MASKS):
        T = S
        while T:
            T = (T - 1) & S                       # enumerate proper subsets of S
            if T and RANK_IDX[T] >= RANK_IDX[S]:
                bad_sub += 1
    # gate (ii): the canonical rep is rank-minimal within every clean orbit
    orbits = {}
    for S in range(1, N_MASKS):
        orbits.setdefault(REP_OF[S], []).append(S)
    bad_rep = sum(1 for rep, mem in orbits.items()
                  if min(mem, key=lambda S: RANK_IDX[S]) != rep)
    # totality
    assert sorted(RANK_IDX[1:]) == list(range(N_MASKS - 1)), "RANK_IDX is not a permutation"
    print(f"topology: {_tc.TOPOLOGY} (N_DEN={_tc.N_DEN}, {N_MASKS - 1} sectors)")
    print(f"gate (i)  subsector < parent violations : {bad_sub}")
    print(f"gate (ii) rep-not-minimal orbits        : {bad_rep}")
    print(f"orbits: {len(orbits)}, sectors: {N_MASKS - 1}, rank is total: OK")
    print("ALL PASS" if bad_sub == 0 and bad_rep == 0 else "FAIL")
