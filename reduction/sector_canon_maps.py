#!/usr/bin/env python
"""STEP 1 of the staged symmetry routing (ORDERING.md design): one composite
sector-changing map per non-canonical sector, sending EVERY integral of that sector
(any dots, any numerators — power-independence lemma) to its canonical sector:
  I  =  (same-(r,s) terms in the canonical sector)  +  (strictly lower debris).

Built once by BFS on the clean corner graph (corner -> corner single-integral edges,
coefficient 1) toward the canonical representative, composing the affine maps along
the path. Cached in results/sector_canon_maps.pkl = {mask: (M, c)} for the 81
non-canonical sectors. Run this file directly to (re)build and gate:
  - every composite maps corner(S) EXACTLY to corner(rep(S));
  - every non-canonical sector has a map (reachability = 255/255, verified 2026-07-10
    in analysis/verify_sector_canonicalization.py).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from canonicalize import _transforms, image_unsigned, P, N

_PKL = os.path.join(ROOT, "results/sector_canon_maps.pkl")


def _compose(A, B):
    """Affine composite: apply A then B on D-space (D -> M D + c over GF(P))."""
    AM, Ac = A; BM, Bc = B
    NM = {}
    for i in range(N):
        acc = {}
        for j, aij in AM.get(i, {}).items():
            for k, bjk in BM.get(j, {}).items():
                acc[k] = (acc.get(k, 0) + aij * bjk) % P
        NM[i] = {k: v for k, v in acc.items() if v % P}
    Nc = {}
    for i in range(N):
        val = Ac.get(i, 0)
        for j, aij in AM.get(i, {}).items():
            val = (val + aij * Bc.get(j, 0)) % P
        Nc[i] = val % P
    return (NM, Nc)


_ID = ({i: {i: 1} for i in range(N)}, {})


def _corner(mask):
    return tuple(1 if mask >> k & 1 else 0 for k in range(8)) + (0, 0, 0)


def _sec(t):
    return sum(1 << k for k in range(8) if t[k] > 0)


def build():
    from canonical_rep import canonical_rep
    rep_mask = {m: _sec(canonical_rep(_corner(m))) for m in range(1, 256)}
    # corner graph edges
    edges = {}
    for m in range(1, 256):
        out = []
        for (M, c) in _transforms(_corner(m)):
            img = image_unsigned(_corner(m), M, c)
            if img is None or len(img) != 1:
                continue
            (J, co), = img.items()
            if co % P != 1:
                continue
            out.append((_sec(J), (M, c)))
        edges[m] = out
    maps = {}
    for m in range(1, 256):
        tgt = rep_mask[m]
        if m == tgt:
            continue
        seen = {m: None}; frontier = [m]; found = False
        while frontier and not found:
            nf = []
            for u in frontier:
                for (v, mc) in edges[u]:
                    if v not in seen:
                        seen[v] = (u, mc); nf.append(v)
                        if v == tgt:
                            found = True; break
                if found:
                    break
            frontier = nf
        if not found:
            raise RuntimeError(f"sector {m}: no clean path to rep {tgt} — record set changed?")
        path = []; v = tgt
        while seen[v] is not None:
            u, mc = seen[v]; path.append(mc); v = u
        g = _ID
        for mc in reversed(path):
            g = _compose(g, mc)
        maps[m] = g
    return maps, rep_mask


def load():
    """Cached load; rebuild if missing."""
    if os.path.exists(_PKL):
        with open(_PKL, "rb") as f:
            return pickle.load(f)["maps"]
    maps, _ = build()
    with open(_PKL, "wb") as f:
        pickle.dump({"maps": maps}, f)
    return maps


if __name__ == "__main__":
    from sailir import ibp_env
    from sailir.topology import Topology
    ibp_env.init_from_topology(Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox")))
    ibp_env.set_prime(1009)
    maps, rep_mask = build()
    with open(_PKL, "wb") as f:
        pickle.dump({"maps": maps}, f)
    bad = 0
    for m, g in maps.items():
        img = image_unsigned(_corner(m), g[0], g[1])
        ok = img is not None and len(img) == 1 and next(iter(img)) == _corner(rep_mask[m])
        if not ok:
            bad += 1; print(f"  GATE FAIL sector {m}: corner image {img}")
    print(f"non-canonical sectors with a composite map: {len(maps)} (expect 81)")
    print(f"corner-exactness gate: {len(maps) - bad}/{len(maps)}")
    print(f"saved -> {_PKL}")
    print("ALL PASS" if bad == 0 and len(maps) == 81 else "FAIL")
