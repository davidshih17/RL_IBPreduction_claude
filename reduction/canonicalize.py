#!/usr/bin/env python
"""Confluent, value-preserving canonicalize() under sector symmetry (GF(p)).

ONE global reduced relation set: union-close the seed integrals under every
applicable (all-supersector) symmetry, form the exact relations K - image(K) = 0
(image = SymmetryReducer._image, the Kira-validated transform), RREF them once
(back-substituted so rule RHS are pivot-free), and reduce any integral against
that fixed map. A fixed GF(p) linear map => confluent + idempotent. Every rule is
a linear combination of EXACT symmetry identities => canon(I) == I (value-preserving);
integrals that vanish are genuine symmetry-zeros; it is the COMPLETE symmetry
reduction (strictly stronger than the permutation-only orbit-min).

Self-test verifies: (V) every rule reduces to 0 under an INDEPENDENT echelon basis
of the raw relations (no spurious relation -> value-preserving); (C) confluence;
(I) idempotency.
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir_symmetry import SymmetryReducer
from sailir.symmetries import sector_of
from symmetry_engine import N
P = 1009
RED = SymmetryReducer(prime=P)
rep_of = pickle.load(open(ROOT + "/results/canonical_sectors.pkl", "rb"))["rep_of"]

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

def elim_key(I):
    S = sector_of(I); deg = sum(-x for x in I if x < 0)
    return (0 if rep_of[S] == S else 1, deg, S, I)

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

def build(seeds, cap=40000):
    seen = set(seeds); frontier = list(seeds); raw = []
    while frontier:
        K = frontier.pop()
        for (M, c) in _transforms(K):
            img = image_unsigned(K, M, c)
            if img is None: continue
            rel = dict(img); rel[K] = (rel.get(K, 0) - 1) % P
            rel = {k: v for k, v in rel.items() if v % P}
            if rel: raw.append(rel)
            for J in img:
                if J not in seen:
                    if len(seen) >= cap: raise RuntimeError("cap")
                    seen.add(J); frontier.append(J)
    pos = {v: k for k, v in enumerate(sorted(seen, key=elim_key))}
    rules = {}
    for rel in raw:
        r = _reduce(rel, rules)
        if not r: continue
        piv = max(r, key=lambda v: pos[v]); co = r.pop(piv); inv = pow(co, P - 2, P)
        newp = {k: (-v * inv) % P for k, v in r.items()}
        for q in list(rules):
            if piv in rules[q]:
                cc = rules[q].pop(piv)
                for w, cw in newp.items(): rules[q][w] = (rules[q].get(w, 0) + cc * cw) % P
                rules[q] = {k: v for k, v in rules[q].items() if v % P}
        rules[piv] = newp
    return rules, raw, seen, pos

def canon(I, rules): return _reduce({tuple(I): 1}, rules)


class Canonicalizer:
    """Stateful, incremental canonicalize for a live reduction. Maintains ONE
    global reduced rule set; pivots by the GLOBAL elim_key (so the canonical basis
    is order-independent -> confluent regardless of the order integrals appear).
    canon(I) lazily extends the closure with I's symmetry relations, then reduces.
    Value-preserving (rules are GF(p) combinations of exact symmetry identities)."""
    def __init__(self):
        self.rules = {}            # pivot integral -> {free integral: coeff}
        self.contains = {}         # free integral -> set of pivots q with it in rules[q]
        self.seen = set()
        self._memo = {}
        self._extend_fires = 0     # count lazy-closure fires (uncovered integrals)
        self._skipped_uncovered = 0
        # SAILIR_CANON_NO_EXTEND=1: do NOT build the closure on the fly for integrals
        # the precompute didn't cover. Instead treat them as their own canonical form
        # (canon(I) = _reduce vs the fixed precomputed rules; if I isn't a pivot there,
        # it stays as I). Value-preserving — you simply forgo the symmetry dedup for
        # out-of-range integrals rather than pay the expensive lazy closure. Graceful
        # degradation: use symmetry where it's cheap (precomputed), skip it where it's
        # not, instead of slowing the whole reduction to a crawl.
        self._no_extend = os.environ.get('SAILIR_CANON_NO_EXTEND', '0') == '1'
        # Load a precomputed global rule set (offline `build`) if provided. Then
        # canon(I) is just _reduce for every covered integral — no closure building
        # in the hot loop. _extend still fires (and stays correct) for anything the
        # precompute didn't cover; the counter reports how often that happens.
        _pf = os.environ.get('SAILIR_CANON_RULES')
        if _pf and os.path.exists(_pf):
            with open(_pf, 'rb') as f:
                d = pickle.load(f)
            self.rules = d['rules']
            self.seen = d['seen']
            print(f"[canon] loaded precomputed rules: {len(self.rules)} rules, "
                  f"{len(self.seen)} closure integrals from {_pf}", flush=True)
        # Reverse index: which rules' RHS contain a given integral. Lets _add
        # back-substitute a new pivot into ONLY the rules that reference it, instead
        # of scanning all (66k+) rules — the difference between O(1) and O(rules) per
        # uncovered integral, i.e. between fast and hanging on the lazy fallback.
        for q, rhs in self.rules.items():
            for k in rhs:
                self.contains.setdefault(k, set()).add(q)

    def _add(self, rel):           # incremental RREF, pivot = max global elim_key, back-substitute
        r = _reduce(rel, self.rules)
        if not r:
            return
        piv = max(r, key=elim_key); co = r.pop(piv); inv = pow(co, P - 2, P)
        newp = {k: (-v * inv) % P for k, v in r.items()}
        # back-substitute piv -> newp only in rules that actually contain piv
        qs = self.contains.pop(piv, None)
        if qs:
            for q in qs:
                row = self.rules.get(q)
                if row is None or piv not in row:
                    continue
                cc = row.pop(piv)
                for w, cw in newp.items():
                    old = row.get(w, 0)
                    nv = (old + cc * cw) % P
                    if nv:
                        if old == 0:
                            self.contains.setdefault(w, set()).add(q)
                        row[w] = nv
                    elif old:
                        del row[w]
                        s = self.contains.get(w)
                        if s is not None:
                            s.discard(q)
        self.rules[piv] = newp
        for k in newp:
            self.contains.setdefault(k, set()).add(piv)

    def _extend(self, I):
        if I in self.seen:
            return
        self._extend_fires += 1
        frontier = [I]; self.seen.add(I); grew = False
        while frontier:
            K = frontier.pop()
            for (M, c) in _transforms(K):
                img = image_unsigned(K, M, c)
                if img is None:
                    continue
                rel = dict(img); rel[K] = (rel.get(K, 0) - 1) % P
                rel = {k: v for k, v in rel.items() if v % P}
                if rel:
                    self._add(rel); grew = True
                for J in img:
                    if J not in self.seen:
                        self.seen.add(J); frontier.append(J)
        if grew:
            self._memo.clear()

    def canon(self, I):
        I = tuple(I)
        m = self._memo.get(I)
        if m is not None:
            return m
        if self._no_extend:
            # Out-of-precompute integrals are NOT closed on the fly: reduce against the
            # fixed precomputed rules only (I stays as I if it isn't a covered pivot).
            if I not in self.seen:
                self._skipped_uncovered += 1
        else:
            self._extend(I)
        r = _reduce({I: 1}, self.rules)
        self._memo[I] = r
        return r

    def canon_sol(self, sol):
        """Canonicalize a substitution/expression {integral: coeff} -> canonical {integral: coeff}."""
        out = {}
        for J, co in sol.items():
            for K, ck in self.canon(J).items():
                out[K] = (out.get(K, 0) + co * ck) % P
        return {k: v for k, v in out.items() if v % P}


class LocalCanonicalizer:
    """Same canonical map as Canonicalizer, but computed PER-ORBIT instead of via one
    global growing RREF.

    The relations K - image(K) = 0 only ever connect symmetry-equivalent integrals, so
    the relation graph is a DISJOINT UNION of finite orbits (connected components). The
    canonical form of I therefore depends only on I's own component — never on any other
    integral. canon(I) builds just that component, RREFs it once (bounded by the orbit
    size, independent of how many integrals have been seen), and memoizes the WHOLE
    component. Cost is per-orbit, not quadratic in the total integral count — which is
    exactly why the global Canonicalizer's `_extend`/`_add` blew up when a step touched
    ~1000 fresh integrals. Output is identical to Canonicalizer (same elim_key pivoting,
    same disjoint components)."""
    def __init__(self):
        self._memo = {}

    def canon(self, I):
        I = tuple(I)
        m = self._memo.get(I)
        if m is not None:
            return m
        rules, _raw, seen, _pos = build([I])
        for J in seen:
            self._memo[J] = _reduce({J: 1}, rules)
        return self._memo[I]

    def canon_sol(self, sol):
        out = {}
        for J, co in sol.items():
            for K, ck in self.canon(J).items():
                out[K] = (out.get(K, 0) + co * ck) % P
        return {k: v for k, v in out.items() if v % P}


if __name__ == "__main__":
    def present(s): return [i for i in range(8) if (s >> i) & 1]
    seeds = []
    for sec in range(1, 1 << 8):
        ps = present(sec)
        if not (1 <= len(ps) <= 3): continue
        b = [0] * N
        for g in ps: b[g] = 1
        seeds.append(tuple(b))                                   # deg 0
        ab = [i for i in range(8) if i not in ps]
        for s in (8, 9, 10) + tuple(ab[:1]):                     # deg 1 numerators
            t = list(b); t[s] = -1; seeds.append(tuple(t))
        t = list(b); t[8] = -2; seeds.append(tuple(t))           # deg 2
    seeds = list(dict.fromkeys(seeds))
    print(f"seeds={len(seeds)}  building global rule set ...", flush=True)
    rules, raw, seen, pos = build(seeds)
    print(f"closure={len(seen)}  rules={len(rules)}  raw_relations={len(raw)}", flush=True)

    # (V) independent echelon of RAW relations; every rule must reduce to 0 under it
    ech = {}                                                     # pivot -> normalized row
    def ech_reduce(d):
        d = dict(d)
        while True:
            hit = next((v for v in d if v in ech), None)
            if hit is None: return {k: x for k, x in d.items() if x % P}
            co = d.pop(hit)
            for w, cw in ech[hit].items(): d[w] = (d.get(w, 0) - co * cw) % P
    for rel in raw:
        r = ech_reduce(rel)
        if not r: continue
        piv = max(r, key=lambda v: pos[v]); co = r.pop(piv); inv = pow(co, P - 2, P)
        ech[piv] = {k: (v * inv) % P for k, v in r.items()}
    Vbad = 0
    for piv, rhs in rules.items():
        d = dict(rhs); d[piv] = (d.get(piv, 0) - 1) % P          # (RHS - piv) should be a true relation
        if ech_reduce(d): Vbad += 1
    # (C)/(I)
    idem = conf = checks = 0
    for I in seeds:
        cI = canon(I, rules)
        ce = {}
        for J, co in cI.items():
            for K, ck in canon(J, rules).items(): ce[K] = (ce.get(K, 0) + co * ck) % P
        if {k: v for k, v in ce.items() if v % P} != cI: idem += 1
        for (M, c) in _transforms(I):
            img = image_unsigned(I, M, c)
            if img is None or img == {I: 1}: continue
            e = {}
            for J, co in img.items():
                for K, ck in canon(J, rules).items(): e[K] = (e.get(K, 0) + co * ck) % P
            checks += 1
            if {k: v for k, v in e.items() if v % P} != cI: conf += 1
            break
    print("=" * 72)
    print(f"(V) VALUE-PRESERVING: rules that are NOT a true symmetry relation : {Vbad} / {len(rules)}")
    print(f"(C) CONFLUENT  canon(sym(I))==canon(I)                            : {conf} fail / {checks}")
    print(f"(I) IDEMPOTENT canon(canon(I))==canon(I)                          : {idem} fail / {len(seeds)}")
    free = len(seen) - len(rules)
    print(f"    free (canonical-basis) integrals in closure                  : {free}")
    for I in [(1,0,0,0,0,1,0,1,0,0,0), (1,0,0,0,0,1,0,1,-1,0,0), (0,0,1,0,1,0,0,1,-1,0,0)]:
        print(f"    canon({I}) = {canon(I, rules)}")
    ok = (Vbad == 0 and conf == 0 and idem == 0)
    print("=" * 72)
    print("RESULT:", "ALL PASS — canonicalize is confluent, idempotent, value-preserving." if ok
          else "FAIL — see above.")
