#!/usr/bin/env python
"""Sector-symmetry transform primitives (GF(p)) for the 174 clean-orbit
canonicalization and the inference routing.

Exposes the value-relation image of an integral under a sector symmetry:
  image_unsigned(a, M, c) = the combination sigma.I(a) (NO Kira determinant sign),
built from the GLOBALLY-CONSISTENT momentum engine (derive_transform), NOT Kira's
per-sector raw matrices. A DENOMINATOR that maps to a COMBINATION makes the whole
symmetry inapplicable (returns None) — this is exactly the clean-denominator gate
that gives the **174** (not the erroneous min-integer 139) equivalence the
successful symmetry inference uses.

_transforms(K) yields the (M, c) transforms applicable to K (all supersector syms);
image_unsigned applies one; _reduce reduces a dict against a rule map. The ordering /
pivot choice for canonicalization lives in the consumers (canonical_rep.py,
symmetry_route.py), which pivot by the workers' _target_key.
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir_symmetry import SymmetryReducer
from sailir.symmetries import sector_of
from symmetry_engine import N
P = 1009
RED = SymmetryReducer(prime=P)

# Build the transform source from the GLOBALLY-CONSISTENT momentum engine
# (derive_transform), NOT Kira's per-sector raw matrices (whose numerator rows are
# sector-local padding and fabricate false relations on subsector numerator
# integrals). Placeholder records (no loop_subst) -> denominator permutation only.
def _build_engine_src():
    # Cache the engine-derived transforms (sympy derive is ~15s) keyed by (prime,kin).
    _ck = os.path.join(ROOT, "reduction", "canon_engine_src.pkl")
    _key = (P, tuple(sorted(RED.kin.items())))
    if os.path.exists(_ck):
        with open(_ck, "rb") as f:
            d = pickle.load(f)
        if d.get("key") == _key:
            return d["src"]
    import sympy as sp
    from sailir.symmetries import parse_symmetries
    from symmetry_engine import derive_transform
    from recon_from_swaps import reconstruct           # true relabeling from clean swaps
    TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
    kin = RED.kin; syms = {sp.Symbol(k): v for k, v in kin.items()}
    recs = (parse_symmetries(os.path.join(TA, "sectorSymmetries"), N, 2)
            + parse_symmetries(os.path.join(TA, "sectorRelations"), N, 2))
    _props = lambda sec: [g for g in range(8) if sec >> g & 1]
    by = {}
    for r in recs:
        ls = list(r.loop_substs)
        if any(rhs == "placeholder" for _, rhs in ls):
            # Placeholder = genuine relabeling whose loop_subst Kira did not store.
            # Reconstruct the TRUE momentum map from the clean swaps (a prop -> a
            # DIFFERENT prop is reliable; "fixed" entries can be fake). The old
            # partial-M (ing permutation) fabricated wrong reductions where a present
            # prop actually maps to a COMBINATION -> reject those via the full M.
            rec = reconstruct(r.ing, _props(r.source_sector))
            if rec is None:
                continue                              # no swaps -> unrecoverable, drop (was: wrong M)
            ls = rec
        Mf, cf = derive_transform(ls)
        M = {i: {j: int(co.subs(syms)) % P for j, co in Mf[i].items()} for i in Mf}
        c = {i: int(cf[i].subs(syms)) % P for i in cf}
        by.setdefault(r.source_sector, []).append((M, c))
    src = list(by.items())
    try:
        with open(_ck, "wb") as f:
            pickle.dump({"key": _key, "src": src}, f)
    except OSError:
        pass
    return src
_SRC = _build_engine_src()

def _transforms(K):
    sK = sector_of(K)
    for S, mcs in _SRC:
        if (S & sK) == sK:
            for mc in mcs:
                yield mc

def image_unsigned(a, M, c):
    """Like SymmetryReducer._image but WITHOUT Kira's determinant sign (the co==-1
    denominator flip), which SAILIR does not apply at the integral-value level.
    The value relation is I(a) = I(sigma.a); applying that spurious sign would make
    a self-symmetry read I = -I => I = 0 (wrong). Numerator/combination coeffs kept."""
    p = P; base = [0] * N; num = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M.get(i)
            if not row or len(row) != 1:
                return None
            j, _co = next(iter(row.items()))
            base[j] += ai                       # NO sign flip
        elif ai < 0:
            row = M.get(i, {})
            if not row and not c.get(i, 0):
                return None
            num.append((-ai, row, c.get(i, 0)))
    res = {tuple(base): 1}
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
            for w, cw in t.items(): out[w] = (out.get(w, 0) + co * cw) % P
        else:
            out[v] = (out.get(v, 0) + co) % P
    return {k: x for k, x in out.items() if x % P}
