#!/usr/bin/env python
"""GRAVITY (GR) canonicalize provider — API parity with canonicalize.py
(_transforms, image_unsigned, _reduce, P, N) so symmetry_route / sector_canon_maps
/ canonical_masters work unchanged via topo_config.canonicalize_module().

Transforms come from results/gr_transforms.pkl (symmetry_engine_GR.py: every Kira
record realized by a verified momentum map + external involutions, den behavior
gated at two evaluation points). Format identical to pentagonbox: per source
sector a list of (M, c), M[i] = {j: coeff}, c[i] = const, all mod P.

ONE BEHAVIORAL DIFFERENCE vs the pentagonbox image function, forced by GR's
LINEAR (eikonal) denominators: a denominator can map to coeff * D_j with
coeff != 1 (e.g. -1 under a reflection). The pentagonbox code ignores den-row
coefficients (always +1 there — quadratic propagators). Here:
  1/D_g^a  ->  1/(co*D_j)^a  =  inv(co)^a / D_j^a,
so the image carries the prefactor inv(co)^a. Additionally a denominator row
must have NO constant part and must land in a DENOMINATOR slot; otherwise the
transform is inapplicable to that integral (returns None) — skipping a symmetry
is always safe. Kira's det sign remains unapplied (same convention as
pentagonbox; it is Jacobian bookkeeping, not a value-level sign).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import sector_of

P = 1009
N = 15
N_DEN = 10

_PKL = os.path.join(ROOT, "results/gr_transforms.pkl")
with open(_PKL, "rb") as f:
    _STORE = pickle.load(f)
assert _STORE.get("prod_point", (P,))[0] == P, "gr_transforms.pkl prime mismatch"
_SRC = sorted(_STORE["by_sector"].items())


def _transforms(K):
    """All transforms applicable by super-sector: source sector S covers K's
    sector sK when (S & sK) == sK (same convention as pentagonbox)."""
    sK = sector_of(K)
    for S, mcs in _SRC:
        if (S & sK) == sK:
            for mc in mcs:
                yield mc


def image_unsigned(a, M, c):
    """Value image of integral `a` under transform (M, c); None if inapplicable.
    Den rows: single-term, no constant, den-slot target; coefficient applied as
    inv(co)^power. Numerator rows: full multinomial expansion with coefficients
    (identical to pentagonbox). Kira det sign NOT applied."""
    p = P
    base = [0] * N
    pref = 1
    num = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M.get(i)
            if not row or len(row) != 1 or c.get(i, 0) % p:
                return None
            j, co = next(iter(row.items()))
            if j >= N_DEN:
                return None                    # den landing on an ISP slot: skip
            base[j] += ai
            if co % p != 1:
                pref = pref * pow(pow(co, p - 2, p), ai, p) % p
        elif ai < 0:
            row = M.get(i, {})
            if not row and not c.get(i, 0):
                return None
            num.append((-ai, row, c.get(i, 0)))
    res = {tuple(base): pref}
    for power, row, const in num:
        for _ in range(power):
            new = {}
            for integ, co in res.items():
                for j, mij in row.items():
                    ni = list(integ); ni[j] -= 1; ni = tuple(ni)
                    new[ni] = (new.get(ni, 0) + co * mij) % p
                if const:
                    new[integ] = (new.get(integ, 0) + co * const) % p
            res = {k: v for k, v in new.items() if v % p}
    return res


def _reduce(d, rules):
    out = {}
    for v, co in d.items():
        t = rules.get(v)
        if t is not None:
            for w, cw in t.items():
                out[w] = (out.get(w, 0) + co * cw) % P
        else:
            out[v] = (out.get(v, 0) + co) % P
    return {k: x for k, x in out.items() if x % P}


if __name__ == "__main__":
    # smoke gates: transform counts; corner self-consistency (every stored map,
    # applied to its source sector's corner, must yield a single corner with a
    # NONZERO coefficient in a valid sector)
    n_maps = sum(len(v) for _, v in _SRC)
    print(f"transform store: {len(_SRC)} sectors, {n_maps} transforms")
    bad = 0
    checked = 0
    for S, mcs in _SRC:
        corner = tuple(1 if S >> i & 1 else 0 for i in range(N_DEN)) + (0,) * (N - N_DEN)
        for (M, c) in mcs:
            img = image_unsigned(corner, M, c)
            checked += 1
            if img is None:
                bad += 1
                continue
            if len(img) != 1:
                bad += 1
                print(f"  corner of {S}: multi-term image {img}")
    print(f"corner applications: {checked}, failures: {bad}")
    print("ALL PASS" if bad == 0 else "FAIL")
