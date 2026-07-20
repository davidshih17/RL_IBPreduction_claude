#!/usr/bin/env python
"""CANONICAL MASTERS — the master basis of the canonical-sector pipeline.

The paper basis (Kira for pentagonbox, FIRE for gravity3L) is defined in the
PAPER's preferred sectors; our pipeline canonicalizes sectors by OUR order
(ORDERING.md). Where the two conventions disagree, the orbit's merged corner is
not a paper master and no worker can reduce it (IBP cannot leave the sector
cone; the identity to the paper master ascends the order) — the m1/m3 hang.

DECISION (dshih, 2026-07-11): transform the paper masters INTO our canonical
sectors using the symmetries — the self-consistent choice; the ordering and
canonicalization stay self-contained and the paper basis is reached through an
exact dictionary at output time.

TOPOLOGY-KEYED (SAILIR_TOPOLOGY via topo_config). Coefficients: pentagonbox
images are all coefficient 1; gravity3L's LINEAR (eikonal) denominators can
give single-term images with coefficient != 1 (reflections), so the dictionary
stores  {canonical_master: (paper_master, co)}  with  I[paper] = co * I[canonical]
(identity entries excluded). translate_to_kira applies inv-co... no: see its
docstring — the stored co is exactly image_unsigned's, and the translation of a
final expression multiplies each canonical-master coefficient by co.

Requires every moved master's image to be a SINGLE integral; raises loudly
otherwise so a genuine multi-term case is a conscious scheme extension.

apply_canonical_masters() swaps ibp_env's master basis in place. Call it AFTER
init_from_topology, in BOTH the orchestrator and every worker, whenever
SAILIR_SECTOR_RANK=1 (one package with the sector-senior order).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc


def build():
    """Compute (canonical_master_set, dictionary) from the current ibp_env
    master basis. Dictionary values:
      (paper, co)                    I[paper] = co * I[canonical]
      (paper, co, debris)            I[paper] = co * I[canonical]
                                       + sum debris[J] * I[J],  every J in a
                                     STRICTLY LOWER sector (reducible, non-master)
    The debris form arises when a paper master carries a numerator power and the
    relabeling expands it (gravity3L: exactly one FIRE master, sector 895 ->
    767, I[m] = J3 - 1/2 J1 + 1/2 J2 with J1/J2 subsector corners). The
    canonical-sector SAME-RANK term must be unique — raises otherwise."""
    from sailir import ibp_env
    _cz = _tc.canonicalize_module()
    import sector_canon_maps
    with open(_tc.CANON_PKL, "rb") as f:
        canon = set(pickle.load(f)["canonical"])
    maps = sector_canon_maps.load()
    P = _cz.P

    def _sec(t):
        return sum(1 << k for k in range(_tc.N_DEN) if t[k] > 0)

    # Pass 1: native canonical-sector masters keep their identity.
    native = {tuple(m) for m in ibp_env.MASTERS_SET if _sec(m) in canon}
    new_set = set(native)
    dictionary = {}
    merges = {}       # dropped_paper: (kept_canonical, co, debris) — the paper
    #                   basis is REDUNDANT under the full symmetry group when a
    #                   moved master's image coincides with an existing master
    #                   (gravity3L: 23 of 24 — FIRE-68 is 45-dimensional modulo
    #                   symmetry). Exact identities, kept for oracle comparison.
    for m in ibp_env.MASTERS_SET:
        S = _sec(m)
        if S in canon:
            continue
        M, c = maps[S]
        img = _cz.image_unsigned(tuple(m), M, c)
        if img is None or not img:
            raise RuntimeError(f"master {list(m)} (sector {S}) has no canonical "
                               f"image: {img} — extend the canonical-masters "
                               f"scheme before proceeding")
        # split image: same-rank terms in the canonical target sector vs
        # strictly-lower-sector debris
        nb_S = bin(S).count("1")
        main = [(J, co) for J, co in img.items()
                if bin(_sec(J)).count("1") == nb_S]
        debris = {J: co % P for J, co in img.items()
                  if bin(_sec(J)).count("1") < nb_S}
        if len(main) + len(debris) != len(img):
            raise RuntimeError(f"master {list(m)} (sector {S}): image has "
                               f"HIGHER-rank terms: {img}")
        if len(main) > 1:
            # numerator-carrying master whose relabel expands: designate the
            # coefficient-1 DIRECT RELABEL as the master; every other term
            # (same-rank included) goes into the debris entry of the output
            # dictionary and must be supplied via debris_reductions at
            # translation time.
            direct = [(J, co) for J, co in main if co % P == 1]
            if len(direct) == 1:
                (j, co), = direct
            else:
                # no unique coefficient-1 relabel: designate the
                # lexicographically smallest same-rank term (deterministic);
                # the dictionary carries its coefficient exactly
                j, co = sorted(main)[0]
            for J, cJ in main:
                if J != j:
                    debris[J] = cJ % P
            print(f"  [canonical_masters] {list(m)} -> designated relabel "
                  f"{list(j)}; {len(main) - 1} same-rank term(s) moved to "
                  f"debris", flush=True)
        elif len(main) == 0:
            raise RuntimeError(f"master {list(m)} (sector {S}): no same-rank "
                               f"canonical image term: {img}")
        else:
            (j, co), = main
        if co % P == 0:
            raise RuntimeError(f"master {list(m)} (sector {S}): zero-coefficient "
                               f"main image term — record set inconsistent")
        if _sec(j) not in canon:
            raise RuntimeError(f"master {list(m)} (sector {S}): main image term "
                               f"{list(j)} not in a canonical sector")
        entry = (tuple(m), co % P) if not debris else (tuple(m), co % P, debris)
        if j in new_set:
            # image coincides with an existing (native or earlier-moved) master:
            # the paper basis is redundant — record the exact merge identity
            merges[tuple(m)] = (j,) + entry[1:]
            continue
        new_set.add(j)
        dictionary[j] = entry
    return frozenset(new_set), dictionary, merges


_applied = None
MERGES = None     # {dropped_paper: (kept_canonical, co[, debris])} after apply —
#                   I[dropped_paper] = co * I[kept] (+ debris); used by oracle-
#                   comparison tooling to collapse the redundant paper basis.


def apply_canonical_masters():
    """Swap ibp_env's master basis to the canonical-sector images (idempotent).
    Returns the dictionary {canonical_master: (paper_master, co[, debris])} for
    output translation; merge identities land in the module global MERGES."""
    global _applied, MERGES
    if _applied is None:
        from sailir import ibp_env
        new_set, dictionary, merges = build()
        ibp_env.set_masters(new_set)
        _applied = dictionary
        MERGES = merges
    return _applied


def translate_to_kira(expr, dictionary, debris_reductions=None):
    """Rename canonical masters back to the paper basis in a final expression.
    I[paper] = co * I[canonical] (+ debris)  =>  a term  x * I[canonical]  becomes
    x*inv(co) * I[paper]  -  x*inv(co) * (debris terms). Debris integrals are
    NON-master (strictly lower sector); their reductions to CANONICAL masters
    must be supplied via debris_reductions = {debris_integral: {canonical_master:
    coeff}} (produced by reducing them with this same pipeline — they are cheap
    subsector corners). Falls back to _tc-topology file
    results/<topology>_debris_reductions.pkl if present. Raises if a needed
    debris reduction is missing — never emits a non-master integral silently.
    Exact identities; output-time only — never a cache entry."""
    _cz = _tc.canonicalize_module()
    P = _cz.P
    if debris_reductions is None:
        _p = os.path.join(ROOT, f"results/{_tc.TOPOLOGY}_debris_reductions.pkl")
        debris_reductions = {}
        if os.path.exists(_p):
            with open(_p, "rb") as f:
                debris_reductions = pickle.load(f)
    # first pass: expand canonical masters into paper masters + debris-in-
    # canonical-master form; second pass translates those canonical masters too
    # (debris reductions land in lower sectors, so one extra pass suffices —
    # asserted below).
    pending = dict(expr)
    out = {}
    for _pass in range(2):
        nxt = {}
        for k, v in pending.items():
            d = dictionary.get(k)
            if d is None:
                out[k] = (out.get(k, 0) + v) % P
                continue
            paper, co = d[0], d[1]
            inv = pow(co, P - 2, P)
            out[paper] = (out.get(paper, 0) + v * inv) % P
            if len(d) == 3:
                for J, cj in d[2].items():
                    red = debris_reductions.get(J)
                    if red is None:
                        raise RuntimeError(
                            f"canonical master {list(k)} needs the reduction of "
                            f"debris integral {list(J)} to translate to the "
                            f"paper basis — reduce it with this pipeline and "
                            f"store it in debris_reductions "
                            f"(results/{_tc.TOPOLOGY}_debris_reductions.pkl)")
                    for Jm, cm in red.items():
                        nxt[Jm] = (nxt.get(Jm, 0) - v * inv * cj) % P
        pending = {k: v for k, v in nxt.items() if v % P}
        if not pending:
            break
    assert not pending or all(dictionary.get(k) is None or len(dictionary[k]) == 2
                              for k in pending), \
        "debris reductions reference affine canonical masters — extend the pass loop"
    for k, v in pending.items():
        d = dictionary.get(k)
        if d is None:
            out[k] = (out.get(k, 0) + v) % P
        else:
            paper, co = d[0], d[1]
            out[paper] = (out.get(paper, 0) + v * pow(co, P - 2, P)) % P
    return {k: v % P for k, v in out.items() if v % P}


if __name__ == "__main__":
    from sailir import ibp_env
    from sailir.topology import Topology
    ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
    ibp_env.set_prime(1009)
    before = set(ibp_env.MASTERS_SET)
    d = apply_canonical_masters()
    after = set(ibp_env.MASTERS_SET)
    with open(_tc.CANON_PKL, "rb") as f:
        canon = set(pickle.load(f)["canonical"])
    bad = [m for m in after
           if sum(1 << k for k in range(_tc.N_DEN) if m[k] > 0) not in canon]
    merges = MERGES or {}      # module global set by apply_canonical_masters()
    print(f"topology: {_tc.TOPOLOGY}")
    print(f"masters before: {len(before)}, after: {len(after)}, "
          f"relabeled: {len(d)}, merged (paper-basis redundancy): {len(merges)}")
    for k, dv in sorted(d.items()):
        extra = f" + debris {[list(j) for j in dv[2]]}" if len(dv) == 3 else ""
        print(f"  canonical {list(k)}  <->  paper {list(dv[0])}  (co={dv[1]}){extra}")
    for m, mv in sorted(merges.items()):
        extra = f" + debris {[list(j) for j in mv[2]]}" if len(mv) == 3 else ""
        print(f"  MERGED paper {list(m)}  ==  {mv[1]} * {list(mv[0])}{extra}")
    print(f"canonical masters in non-canonical sectors: {len(bad)}")
    ok = len(after) == len(before) - len(merges) and not bad
    if _tc.TOPOLOGY == 'pentagonbox':
        I0 = (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0)
        ibp_env.set_paper_masters_only(True)
        print(f"is_master(stuck corner I0) now: {ibp_env.is_master(I0)}")
        ok = ok and len(d) == 1 and not merges and ibp_env.is_master(I0)
    print("ALL PASS" if ok else "FAIL")
