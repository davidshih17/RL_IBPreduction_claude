#!/usr/bin/env python
"""GENERAL symmetry engine — one implementation for every topology.

Everything is derived from the standard topology inputs; NOTHING family-
specific lives in the code:
  <topo>/integralfamilies.yaml   loop momenta, propagator expressions
  <topo>/kinematics.yaml         externals, momentum conservation,
                                 scalar-product rules, invariants
  Kira sectormappings            sectorSymmetries + sectorRelations records
  kinematics values              numeric invariant values (mod p)

Pipeline (validated on gravity3L, 2026-07-17):
  1. scalar-product basis: loop.loop + loop.external pairs; the propagator
     matrix A (n_props x n_sp) must be square + invertible (complete family).
  2. candidate maps: loop relabelings  k_i -> sum_j a_ij k_j + sum_a b_ia e_a
     over the SHIFT vocabulary (externals appearing inside squared propagator
     momenta), entries {-1,0,1}, loop-block det = +-1; composed with the
     EXTERNAL involutions (signed permutations of the externals preserving the
     Gram matrix symbolically — found automatically, e.g. gravity's u1<->u2
     and q->-q; pentagonbox has none at generic kinematics).
  3. transform (M, c): S = S_phi @ R^-1 — the loop-substitution action on the
     sp basis, with the external relabeling applied to the COLUMN basis and
     all constants R-invariant (getting this composition wrong produced false
     I = -I zeros in the gravity prototype; the numeric gate caught it).
  4. every Kira record is realized by matching its den permutation (`ing`) on
     the source sector — grid match, or a targeted exact linear solve for
     out-of-grid (coefficient-2) maps; unrealized records fail the gate.
  5. EVERY stored transform passes the NUMERIC GROUND-TRUTH GATE: explicit
     GF(p) vectors realizing the external Gram exactly, random loops, and the
     propagators evaluated from their parsed momentum-monomial definitions
     (fully independent of the A-matrix machinery):
        D_g(phi(k); e)  ==  sum_h M[g][h] D_h(k; e') + c_g .

Consumers (canonicalize2) apply the empirically locked CLEAN-DEN convention:
a denominator row is usable only with coefficient exactly +1 and no constant —
flip maps are NOT value identities for eikonal (linear) denominators
(analysis/test_merge_signs_vs_fire.py: 677 FIRE-pair comparisons, unanimous).

Usage:
  symmetry_engine2.py <topology_dir> <sectormappings_dir> <out_pkl> [prime]
"""
import os, re, sys, itertools, pickle, random
from fractions import Fraction
import numpy as np
import sympy as sp

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)

CHECK_PRIME = 2003


# ---------------------------------------------------------------------------
# topology input parsing (no pyyaml on this cluster — regex/line based, same
# spirit as sailir/topology.py)
# ---------------------------------------------------------------------------
def parse_kinematics_yaml(path):
    txt = open(path).read()

    def lst(key):
        m = re.search(rf"{key}\s*:\s*\[([^\]]*)\]", txt)
        return [t.strip().strip('"') for t in m.group(1).split(",")
                if t.strip()] if m else []

    invariants = re.findall(r"-\s*\[\s*(\w+)\s*,\s*\d+\s*\]", txt)
    rules = []
    for m in re.finditer(r"-\s*\[\[([^\]]+)\]\s*,\s*([^\]\n]+?)\]\s*$", txt, re.M):
        e1, e2 = [t.strip() for t in m.group(1).split(",")]
        rules.append((e1, e2, m.group(2).strip()))
    return dict(incoming=lst("incoming_momenta"), outgoing=lst("outgoing_momenta"),
                conservation=lst("momentum_conservation"),
                invariants=invariants, rules=rules)


def parse_integralfamilies_yaml(path):
    txt = open(path).read()
    loops = [t.strip() for t in
             re.search(r"loop_momenta\s*:\s*\[([^\]]*)\]", txt).group(1).split(",")]
    props = [(e, m.strip('"')) for e, m in
             re.findall(r'-\s*\[\s*"([^"]+)"\s*,\s*([^\]\s]+)\s*\]', txt)]
    m = re.search(r"top_level_sectors\s*:\s*\[([^\]]*)\]", txt)
    top = int(m.group(1).split(",")[0]) if m else (1 << len(props)) - 1
    return loops, props, top


class GeneralEngine:
    """All family structure derived from the two yaml files."""

    def __init__(self, topo_dir, prime, kin_values,
                 check_prime=CHECK_PRIME, check_values=None):
        loops, props_raw, top = parse_integralfamilies_yaml(
            os.path.join(topo_dir, "integralfamilies.yaml"))
        kin = parse_kinematics_yaml(os.path.join(topo_dir, "kinematics.yaml"))
        self.n_loop = len(loops)
        self.N_IND = len(props_raw)
        self.N_DEN = bin(top).count("1")
        assert top == (1 << self.N_DEN) - 1, "non-contiguous denominator block"

        self.loop_syms = [sp.Symbol(l) for l in loops]
        ext_all = kin["incoming"] + kin["outgoing"]
        subs = {}
        if kin["conservation"]:
            el = kin["conservation"][0]
            expr = ",".join(kin["conservation"][1:]).strip('"')
            subs[sp.Symbol(el)] = sp.sympify(expr)
            ext_all = [e for e in ext_all if e != el]
        self.ext_syms = [sp.Symbol(e) for e in ext_all]
        self.n_ext = len(self.ext_syms)
        self.inv_names = kin["invariants"]

        # ---- external Gram from the scalar-product rules ----
        ne = self.n_ext
        pairs = [(a, b) for a in range(ne) for b in range(a, ne)]
        self.ext_pairs = pairs
        dsym = {p_: sp.Symbol(f"_ed_{p_[0]}_{p_[1]}") for p_ in pairs}
        eqs = []
        for e1, e2, val in kin["rules"]:
            v1 = sp.expand(sp.sympify(e1).subs(subs))
            v2 = sp.expand(sp.sympify(e2).subs(subs))
            lhs = sp.Integer(0)
            for a, sa in enumerate(self.ext_syms):
                for b, sb in enumerate(self.ext_syms):
                    coef = v1.coeff(sa) * v2.coeff(sb)
                    if coef != 0:
                        lhs += coef * dsym[(min(a, b), max(a, b))]
            eqs.append(sp.Eq(lhs, sp.sympify(val)))
        sol = sp.solve(eqs, list(dsym.values()), dict=True)
        assert len(sol) == 1 and len(sol[0]) == len(pairs), \
            "external Gram under/over-determined by the scalarproduct_rules"
        self.gram = {p_: sp.expand(sol[0][dsym[p_]]) for p_ in pairs}

        # ---- propagators -> momentum-monomial lists ----
        # symbol index space: 0..n_loop-1 loops, n_loop..n_loop+n_ext-1 ext
        self.all_syms = self.loop_syms + self.ext_syms
        self.props = []                 # (monos [(coef, i, j)], const, mom_vec)
        for expr_s, mass in props_raw:
            e = sp.expand(sp.sympify(expr_s).subs(subs))
            mom_vec = None
            if e != 0 and sp.Poly(e, *self.all_syms).total_degree() == 1:
                mom_vec = tuple(int(e.coeff(s)) for s in self.all_syms)
                e = sp.expand(e * e)
            monos, const = [], sp.Integer(0) - sp.sympify(mass)
            for mono, coef in sp.Poly(e, *self.all_syms).terms():
                nz = [k for k, d in enumerate(mono) if d]
                if sum(mono) == 0:
                    const += coef
                elif sum(mono) == 2:
                    i, j = (nz[0], nz[0]) if len(nz) == 1 else (nz[0], nz[1])
                    monos.append((int(coef), i, j))
                else:
                    raise RuntimeError(f"propagator {expr_s}: bad degree")
            self.props.append((monos, const, mom_vec))

        # ---- sp basis ----
        nl = self.n_loop
        self.sp_pairs = ([(i, j) for i in range(nl) for j in range(i, nl)]
                         + [(i, nl + a) for i in range(nl) for a in range(ne)])
        self.sp_idx = {p_: k for k, p_ in enumerate(self.sp_pairs)}
        assert len(self.sp_pairs) == self.N_IND, \
            f"family not complete: {len(self.sp_pairs)} sp unknowns, {self.N_IND} props"

        # ---- shift vocabulary: externals inside squared momenta ----
        sh = set()
        for _, _, mv in self.props:
            if mv is not None:
                for a in range(ne):
                    if mv[nl + a]:
                        sh.add(a)
        self.shift_ext = sorted(sh)

        # ---- external involutions (signed Gram-preserving permutations) ----
        self.ext_variants = []
        for perm in itertools.permutations(range(ne)):
            for signs in itertools.product((1, -1), repeat=ne):
                if all(sp.expand(self.gram[(a, b)] - signs[a] * signs[b]
                                 * self.gram[(min(perm[a], perm[b]),
                                              max(perm[a], perm[b]))]) == 0
                       for (a, b) in pairs):
                    self.ext_variants.append((perm, signs))

        # ---- numeric contexts ----
        if check_values is None:
            check_values = {k: (v * 7 + 13) % check_prime
                            for k, v in kin_values.items()}
        self.ctxP = self._make_ctx(prime, kin_values)
        self.ctxC = self._make_ctx(check_prime, check_values)
        self.prime = prime

    # ---- numeric evaluation of an invariant expression ----
    def _ev(self, expr, p, vals):
        e = sp.sympify(expr)
        for name in self.inv_names:
            e = e.subs(sp.Symbol(name), vals.get(name, 1))
        fr = Fraction(int(sp.nsimplify(e).p), int(sp.nsimplify(e).q)) \
            if isinstance(sp.nsimplify(e), sp.Rational) else None
        assert fr is not None, f"non-rational after substitution: {expr}"
        return fr.numerator * pow(fr.denominator % p, p - 2, p) % p

    def _make_ctx(self, p, vals):
        n, nl = self.N_IND, self.n_loop
        A = np.zeros((n, n), dtype=np.int64)
        a0 = np.zeros(n, dtype=np.int64)
        for g, (monos, const, _) in enumerate(self.props):
            for coef, i, j in monos:
                cv = coef % p
                if i < nl and j < nl:
                    A[g, self.sp_idx[(min(i, j), max(i, j))]] += cv
                elif i < nl or j < nl:
                    li, ea = (i, j - nl) if i < nl else (j, i - nl)
                    A[g, self.sp_idx[(li, nl + ea)]] += cv
                else:
                    a0[g] += cv * self._ev(
                        self.gram[(min(i, j) - nl, max(i, j) - nl)], p, vals)
            a0[g] = (a0[g] + self._ev(const, p, vals)) % p
        A %= p
        return dict(p=p, A=A, Ainv=self._inv_mod(A, p), a0=a0 % p,
                    gram={pr: self._ev(self.gram[pr], p, vals)
                          for pr in self.ext_pairs},
                    vals=vals)

    @staticmethod
    def _inv_mod(Amat, p):
        n = Amat.shape[0]
        M = np.concatenate([Amat % p, np.eye(n, dtype=np.int64)], axis=1)
        for col in range(n):
            piv = next((r for r in range(col, n) if M[r, col] % p), None)
            assert piv is not None, f"matrix singular mod {p}"
            M[[col, piv]] = M[[piv, col]]
            M[col] = (M[col] * pow(int(M[col, col]), p - 2, p)) % p
            for r in range(n):
                if r != col and M[r, col]:
                    M[r] = (M[r] - M[r, col] * M[col]) % p
        return M[:, n:]

    # ---- transform of a candidate map ----
    def sp_action(self, a, b, ctx, variant):
        """S = S_phi @ R^-1; s0 = S_phi constants (R-invariant).
        a[i][l] loop block; b[i][ea] external shifts; variant (perm, signs):
        target-side externals e'_a = signs[a] * e_{perm[a]} — on the COLUMN
        basis, S_phi's (l, e_a) column lands at (l, perm[a]) with signs[a]."""
        p = ctx['p']; gram = ctx['gram']
        nl, ne = self.n_loop, self.n_ext
        perm, signs = variant
        S = np.zeros((self.N_IND, self.N_IND), dtype=np.int64)
        s0 = np.zeros(self.N_IND, dtype=np.int64)
        for i in range(nl):
            for j in range(i, nl):
                r = self.sp_idx[(i, j)]
                for l in range(nl):
                    for m in range(nl):
                        S[r, self.sp_idx[(min(l, m), max(l, m))]] += a[i][l] * a[j][m]
                    for ea in range(ne):
                        S[r, self.sp_idx[(l, nl + perm[ea])]] += signs[ea] * (
                            a[i][l] * b[j][ea] + b[i][ea] * a[j][l])
                acc = 0
                for ea in range(ne):
                    for eb in range(ne):
                        acc += b[i][ea] * b[j][eb] * gram[(min(ea, eb), max(ea, eb))]
                s0[r] = acc % p
            for ea in range(ne):
                r = self.sp_idx[(i, nl + ea)]
                for l in range(nl):
                    S[r, self.sp_idx[(l, nl + perm[ea])]] += signs[ea] * a[i][l]
                acc = 0
                for eb in range(ne):
                    acc += b[i][eb] * gram[(min(ea, eb), max(ea, eb))]
                s0[r] = acc % p
        return S % p, s0

    def transform_of(self, a, b, ctx, variant):
        p = ctx['p']
        S, s0 = self.sp_action(a, b, ctx, variant)
        M = (ctx['A'] @ S @ ctx['Ainv']) % p
        c = (ctx['A'] @ s0 + ctx['a0'] - M @ ctx['a0']) % p
        return M, c

    def den_signature(self, M, c, p):
        sig = np.full(self.N_DEN, -1, dtype=np.int8)
        for g in range(self.N_DEN):
            nz = np.nonzero(M[g] % p)[0]
            if len(nz) == 1 and nz[0] < self.N_DEN and c[g] % p == 0:
                sig[g] = nz[0]
        return sig

    # ---- numeric ground-truth gate ----
    @staticmethod
    def _solve_linear(U, g, p, rng, dim):
        """One solution v of  U v = g (mod p)  with the free coordinates set
        randomly (U: k x dim, k <= dim, full row rank as VECTORS — works for
        any Gram, including null momenta where Gram-based projection fails)."""
        k = len(U)
        M = [list(row) + [int(gi)] for row, gi in zip(U, g)]
        piv = []
        r = 0
        for col in range(dim):
            pr = next((i for i in range(r, k) if M[i][col] % p), None)
            if pr is None:
                continue
            M[r], M[pr] = M[pr], M[r]
            inv = pow(M[r][col], p - 2, p)
            M[r] = [x * inv % p for x in M[r]]
            for i in range(k):
                if i != r and M[i][col]:
                    f = M[i][col]
                    M[i] = [(x - f * y) % p for x, y in zip(M[i], M[r])]
            piv.append(col); r += 1
            if r == k:
                break
        assert r == k, "external vectors linearly dependent"
        v = [rng.randrange(p) for _ in range(dim)]
        for i, col in enumerate(piv):
            v[col] = 0
        for i, col in enumerate(piv):
            v[col] = (M[i][dim] - sum(M[i][c] * v[c] for c in range(dim)
                                      if c not in piv)) % p
        return np.array(v, dtype=np.int64)

    def _realize_externals(self, ctx, rng, dim):
        """Explicit GF(p)^dim vectors (identity bilinear form) with the exact
        external Gram. Per vector: solve the LINEAR dot constraints against the
        previous vectors directly (valid for singular Grams / null momenta),
        then fix the norm by adding t*w with w a random nullspace direction
        (quadratic in t; retry until the discriminant is a square)."""
        p = ctx['p']; gram = ctx['gram']
        vecs = []
        for a in range(self.n_ext):
            done = False
            U = [list(map(int, u)) for u in vecs]
            g = [gram[(min(j, a), max(j, a))] for j in range(len(vecs))]
            for _ in range(300):
                v = self._solve_linear(U, g, p, rng, dim) if U else \
                    np.array([rng.randrange(p) for _ in range(dim)], dtype=np.int64)
                want = gram[(a, a)]
                cur = int((v * v).sum() % p)
                if cur == want:
                    vecs.append(v); done = True; break
                w = self._solve_linear(U, [0] * len(U), p, rng, dim) if U else \
                    np.array([rng.randrange(p) for _ in range(dim)], dtype=np.int64)
                vw = int((v * w).sum() % p)
                ww = int((w * w).sum() % p)
                if ww == 0:
                    continue
                # (v + t w)^2 = want:  t^2 ww + 2 t vw + (cur - want) = 0
                disc = (vw * vw - ww * (cur - want)) % p
                r_ = next((x for x in range(p) if x * x % p == disc), None)
                if r_ is None:
                    continue
                t = (-vw + r_) * pow(ww, p - 2, p) % p
                v = (v + t * w) % p
                if int((v * v).sum() % p) == want:
                    vecs.append(v); done = True; break
            if not done:
                raise RuntimeError(f"external Gram realization failed at {a}")
        return vecs

    def _prop_values(self, loop_vecs, ext_vecs, ctx):
        p = ctx['p']
        allv = list(loop_vecs) + list(ext_vecs)
        out = []
        for monos, const, _ in self.props:
            acc = self._ev(const, p, ctx['vals'])
            for coef, i, j in monos:
                acc = (acc + coef * int((allv[i] * allv[j]).sum() % p)) % p
            out.append(acc)
        return out

    def verify_map(self, a, b, variant, ctx, rng, n_trials=3):
        """True iff (M, c) is an exact identity on explicit vector kinematics:
        D_g(phi(k); e) == sum_h M[g][h] D_h(k; e') + c_g with
        e'_a = signs[a] * e_{perm[a]}. The external vectors are realized once
        per context and cached (map-independent)."""
        p = ctx['p']
        M, c = self.transform_of(a, b, ctx, variant)
        dim = self.n_ext + self.n_loop + 2
        ext = ctx.get('_ext_vecs')
        if ext is None:
            ext = self._realize_externals(ctx, rng, dim)
            ctx['_ext_vecs'] = ext
        perm, signs = variant
        ext_rel = [(signs[a2] * ext[perm[a2]]) % p for a2 in range(self.n_ext)]
        for _ in range(n_trials):
            ks = [np.array([rng.randrange(p) for _ in range(dim)], dtype=np.int64)
                  for _ in range(self.n_loop)]
            D = self._prop_values(ks, ext_rel, ctx)
            kp = [(sum(a[i][j] * ks[j] for j in range(self.n_loop))
                   + sum(b[i][ea] * ext[ea] for ea in range(self.n_ext))) % p
                  for i in range(self.n_loop)]
            Dp = self._prop_values(kp, ext, ctx)
            for g in range(self.N_IND):
                rhs = (sum(int(M[g, j]) * D[j] for j in range(self.N_IND))
                       + int(c[g])) % p
                if Dp[g] % p != rhs:
                    return False
        return True

    # ---- targeted solver for out-of-grid maps ----
    def targeted_search(self, present, ing, variant):
        """Exact linear solve for (a, b) realizing the SWAP entries of `ing`
        (ing[g] != g) under the external variant. Constraints:
          pure-square den g -> h:  phi(m_g) = eps * (m_h at relabeled ext)
          loop-linear den  g -> h:  the loop-sp part matches (external tags
                                    must correspond under the variant)
        Enumerates den sign choices; keeps integer solutions, loop det +-1."""
        nl, ne = self.n_loop, self.n_ext
        perm, signs = variant
        nun = nl * nl + nl * ne
        cons = []
        for g in present:
            h = ing[g]
            if h == g:
                continue      # FIXED entries unconstrained: Kira writes fake
                #               'fixed' for dens that actually map to
                #               combinations (pentagonbox sector 201 verified);
                #               the caller re-checks the full den signature via
                #               match_record + the numeric gate.
            mg, mh = self.props[g][2], self.props[h][2]
            if (mg is None) != (mh is None):
                return []
            cons.append(('sq' if mg is not None else 'lin', g, h))
        if not cons:
            return []
        sols = []
        for eps_v in itertools.product((1, -1), repeat=len(cons)):
            rows, rhs = [], []
            ok = True
            for (kind, g, h), eps in zip(cons, eps_v):
                if kind == 'sq':
                    mg, mh = self.props[g][2], self.props[h][2]
                    # relabeled target momentum: loop coeffs unchanged;
                    # external coeff on e_a comes from mh at e'_x where
                    # e'_x = signs[x] e_{perm[x]}: coeff_a = sum over x with
                    # perm[x]=a of mh[nl+x]*signs[x]
                    mh_ext = [0] * ne
                    for x in range(ne):
                        mh_ext[perm[x]] += mh[nl + x] * signs[x]
                    for j in range(nl):
                        row = [0] * nun
                        for i in range(nl):
                            row[i * nl + j] = mg[i]
                        rows.append(row); rhs.append(eps * mh[j])
                    for ea in range(ne):
                        row = [0] * nun
                        for i in range(nl):
                            row[nl * nl + i * ne + ea] = mg[i]
                        rows.append(row)
                        rhs.append(eps * mh_ext[ea] - mg[nl + ea])
                else:
                    # loop-linear props: monomials (coef, loop, ext). The
                    # image's loop-sp part: sum_i coef_i (sum_l a_il k_l).e'_x.
                    # Build per (l, target-ext a) equations.
                    tg = {}
                    for coef, i, j in self.props[g][0]:
                        li, ea = (i, j - nl) if i < nl else (j, i - nl)
                        tg[(li, ea)] = tg.get((li, ea), 0) + coef
                    th = {}
                    for coef, i, j in self.props[h][0]:
                        li, ea = (i, j - nl) if i < nl else (j, i - nl)
                        th[(li, ea)] = th.get((li, ea), 0) + coef
                    # image of tg under phi + variant, as function of a:
                    # sum_i tg[(i,ea)] a_il on (l, perm[ea]) with signs[ea];
                    # external-shift part of phi contributes constants
                    # (e.e' Gram) — require them to cancel: propagators here
                    # are pure loop-linear, so also constrain b-part to keep
                    # no constant: sum_i tg[(i,ea)] b_i(eb) gram terms == 0 is
                    # NONLINEAR-free (linear in b) — include as equations.
                    for l in range(nl):
                        for ta in range(ne):
                            row = [0] * nun
                            for i in range(nl):
                                for ea in range(ne):
                                    if perm[ea] == ta and (i, ea) in tg:
                                        row[i * nl + l] += tg[(i, ea)] * signs[ea]
                            rows.append(row)
                            rhs.append(eps * th.get((l, ta), 0))
            # solve exactly over Fractions
            A = [[Fraction(x) for x in row] + [Fraction(r)]
                 for row, r in zip(rows, rhs)]
            nr, m = len(A), nun
            piv_cols, r_ = [], 0
            for cidx in range(m):
                piv = next((k for k in range(r_, nr) if A[k][cidx] != 0), None)
                if piv is None:
                    continue
                A[r_], A[piv] = A[piv], A[r_]
                A[r_] = [x / A[r_][cidx] for x in A[r_]]
                for k in range(nr):
                    if k != r_ and A[k][cidx] != 0:
                        f = A[k][cidx]
                        A[k] = [x - f * y for x, y in zip(A[k], A[r_])]
                piv_cols.append(cidx); r_ += 1
            if any(all(A[k][c2] == 0 for c2 in range(m)) and A[k][m] != 0
                   for k in range(r_, nr)):
                continue
            free = [c2 for c2 in range(m) if c2 not in piv_cols]
            if len(free) > 5:
                continue
            for fv in itertools.product((-2, -1, 0, 1, 2), repeat=len(free)):
                x = [Fraction(0)] * m
                for c2, v in zip(free, fv):
                    x[c2] = Fraction(v)
                for k, c2 in enumerate(piv_cols):
                    x[c2] = A[k][m] - sum(A[k][j] * x[j] for j in free)
                if any(v.denominator != 1 for v in x):
                    continue
                xi = [int(v) for v in x]
                a = tuple(tuple(xi[i * self.n_loop:(i + 1) * self.n_loop])
                          for i in range(self.n_loop))
                b = tuple(tuple(xi[nl * nl + i * ne:nl * nl + (i + 1) * ne])
                          for i in range(self.n_loop))
                if abs(round(np.linalg.det(np.array(a, dtype=float)))) == 1:
                    sols.append((a, b))
        return sols


# ---------------------------------------------------------------------------
# store build + gates
# ---------------------------------------------------------------------------
def to_dicts(M, c, p, n):
    Md = {i: {int(j): int(M[i, j] % p) for j in np.nonzero(M[i] % p)[0]}
          for i in range(n)}
    Md = {i: r for i, r in Md.items() if r}
    cd = {i: int(c[i] % p) for i in range(n) if c[i] % p}
    return Md, cd


def build_store(topo_dir, symdir, out_pkl, prime=1009, kin_values=None):
    from sailir.symmetries import parse_symmetries
    from sailir.topology import Topology
    topo = Topology.from_dir(topo_dir)
    if kin_values is None:
        kin_values = dict(topo.kinematics_values)
    eng = GeneralEngine(topo_dir, prime, kin_values)
    print(f"topology {os.path.basename(topo_dir)}: loops={eng.n_loop} "
          f"ext={eng.n_ext} props={eng.N_IND} dens={eng.N_DEN} "
          f"shift-ext={eng.shift_ext} ext-variants={len(eng.ext_variants)}")

    recs = (parse_symmetries(os.path.join(symdir, "sectorSymmetries"),
                             eng.N_IND, n_loops=eng.n_loop)
            + parse_symmetries(os.path.join(symdir, "sectorRelations"),
                               eng.N_IND, n_loops=eng.n_loop))
    print(f"records: {len(recs)}")

    # ---- candidate grid ----
    nl, ne = eng.n_loop, eng.n_ext
    loop_mats = [m for m in itertools.product((-1, 0, 1), repeat=nl * nl)
                 if round(np.linalg.det(
                     np.array(m, dtype=float).reshape(nl, nl))) in (1, -1)]
    shift_opts = list(itertools.product((-1, 0, 1), repeat=len(eng.shift_ext)))
    n_cand = len(loop_mats) * len(shift_opts) * len(eng.ext_variants)
    print(f"grid: {len(loop_mats)} loop maps x {len(shift_opts)} shifts x "
          f"{len(eng.ext_variants)} variants = {n_cand}")

    maps, sig_rows = [], []
    for m in loop_mats:
        a = tuple(tuple(m[i * nl:(i + 1) * nl]) for i in range(nl))
        # per-loop independent shift rows:
        for so in itertools.product(shift_opts, repeat=nl):
            b = tuple(tuple(so[i][eng.shift_ext.index(ea)]
                            if ea in eng.shift_ext else 0
                            for ea in range(ne)) for i in range(nl))
            for var in eng.ext_variants:
                MP, cP = eng.transform_of(a, b, eng.ctxP, var)
                sP = eng.den_signature(MP, cP, eng.ctxP['p'])
                if (sP >= 0).any():
                    MC, cC = eng.transform_of(a, b, eng.ctxC, var)
                    sC = eng.den_signature(MC, cC, eng.ctxC['p'])
                    sP[sP != sC] = -1
                maps.append((a, b, var))
                sig_rows.append(sP)
    sigs = np.array(sig_rows, dtype=np.int8)
    print(f"candidates with >=1 two-point-clean den slot: "
          f"{int(((sigs >= 0).sum(axis=1) > 0).sum())}/{len(maps)}")

    rng = random.Random(20260717)
    by_sector = {}
    stats = dict(real=0, real_fail=0, ph=0, ph_matched=0, ph_targeted=0,
                 unmatched=[], numeric_fail=0)

    def parse_subst(loop_substs):
        """Generic loop_subst -> (a, b) or None for placeholders."""
        loc = {str(s): s for s in eng.all_syms}
        a = [[0] * nl for _ in range(nl)]
        b = [[0] * ne for _ in range(nl)]
        for lhs, rhs in loop_substs:
            if rhs == "placeholder":
                return None
            i = [str(s) for s in eng.loop_syms].index(lhs)
            e = sp.expand(sp.sympify(rhs, locals=loc))
            for j in range(nl):
                a[i][j] = int(e.coeff(eng.loop_syms[j]))
            for ea in range(ne):
                b[i][ea] = int(e.coeff(eng.ext_syms[ea]))
        det = round(np.linalg.det(np.array(a, dtype=float)))
        assert det in (1, -1), f"non-unimodular subst {loop_substs}"
        return tuple(tuple(r) for r in a), tuple(tuple(r) for r in b)

    ID_VAR = (tuple(range(ne)), (1,) * ne)

    def match_record(sig_row, present, ing):
        """Kira-faithful matching: a SWAP entry (ing[g] != g) must be realized
        by a clean den row landing at ing[g]; a FIXED entry (ing[g] == g) is
        satisfied by a clean row to g OR a non-clean row (Kira writes fake
        'fixed' for dens that actually map to combinations — the pentagonbox
        lesson)."""
        for g in present:
            if ing[g] != g:
                if sig_row[g] != ing[g]:
                    return False
            elif sig_row[g] not in (g, -1):
                return False
        return True

    def match_mask(sig_mat, present, ing):
        """Vectorized match_record over a (n_maps, N_DEN) signature matrix."""
        cond = np.ones(len(sig_mat), dtype=bool)
        for g in present:
            if ing[g] != g:
                cond &= (sig_mat[:, g] == ing[g])
            else:
                cond &= (sig_mat[:, g] == g) | (sig_mat[:, g] == -1)
        return cond

    # real-substitution maps join the candidate pool for placeholder matching
    # (they carry shift coefficients — e.g. 2*p1 — the {-1,0,1} grid lacks)
    real_maps = []
    for r in recs:
        ab = parse_subst(list(r.loop_substs))
        if ab is not None:
            real_maps.append((ab[0], ab[1], ID_VAR))
    pool = maps + real_maps
    pool_sigs = np.concatenate([sigs] + [np.stack([
        eng.den_signature(*eng.transform_of(a, b, eng.ctxP, v), eng.ctxP['p'])
        for (a, b, v) in real_maps])] if real_maps else [sigs], axis=0) \
        if real_maps else sigs

    for r in recs:
        present = [g for g in range(eng.N_DEN) if r.source_sector >> g & 1]
        ing = r.ing
        ab = parse_subst(list(r.loop_substs))
        if ab is not None:
            # real substitution: numeric verification is the gate; ing
            # agreement is informational only (Kira's ing can be fake-fixed)
            stats['real'] += 1
            if not eng.verify_map(ab[0], ab[1], ID_VAR, eng.ctxP, rng):
                stats['real_fail'] += 1
                print(f"  REAL NUMERIC FAIL sector {r.source_sector}")
                continue
            MP, cP = eng.transform_of(ab[0], ab[1], eng.ctxP, ID_VAR)
            Md, cd = to_dicts(MP, cP, eng.ctxP['p'], eng.N_IND)
            by_sector.setdefault(r.source_sector, []).append((Md, cd))
        else:
            stats['ph'] += 1
            idx = np.nonzero(match_mask(pool_sigs, present, ing))[0]
            # candidates with their clean-den counts (from the precomputed
            # signatures — no per-candidate transform recomputation)
            cands = [(pool[t],
                      int((pool_sigs[t][present] >= 0).sum())) for t in idx]

            # run the targeted solver unless some grid candidate already
            # realizes EVERY present den cleanly — the solver recovers fully-
            # clean out-of-grid (coefficient-2) maps that beat partially-clean
            # grid matches under the max-clean preference below
            if not cands or max(nc for _, nc in cands) < len(present):
                found = []
                for var in eng.ext_variants:
                    for a, b in eng.targeted_search(present, ing, var):
                        MP, cP = eng.transform_of(a, b, eng.ctxP, var)
                        sP = eng.den_signature(MP, cP, eng.ctxP['p'])
                        MC, cC = eng.transform_of(a, b, eng.ctxC, var)
                        sC = eng.den_signature(MC, cC, eng.ctxC['p'])
                        if match_record(sP, present, ing) and \
                                match_record(sC, present, ing):
                            found.append(((a, b, var),
                                          int((sP[present] >= 0).sum())))
                if found:
                    stats['ph_targeted'] += 1
                cands = cands + found
            if not cands:
                stats['unmatched'].append(
                    (r.source_sector, tuple(int(ing[g]) for g in present)))
                continue
            # keep only the maximally-clean candidates (legacy recon's
            # preference — avoids flooding the store with barely-applicable
            # fake-fixed matches), THEN numeric-verify just those
            best = max(nc for _, nc in cands)
            best_cands = [m3 for m3, nc in cands if nc == best]
            verified = [m3 for m3 in best_cands
                        if eng.verify_map(m3[0], m3[1], m3[2], eng.ctxP, rng)]
            stats['numeric_fail'] += len(best_cands) - len(verified)
            if not verified:
                stats['unmatched'].append(
                    (r.source_sector, tuple(int(ing[g]) for g in present)))
                continue
            stats['ph_matched'] += 1
            for a, b, var in verified:
                MP, cP = eng.transform_of(a, b, eng.ctxP, var)
                Md, cd = to_dicts(MP, cP, eng.ctxP['p'], eng.N_IND)
                by_sector.setdefault(r.source_sector, []).append((Md, cd))

    total = 0
    for s in list(by_sector):
        seen, out = set(), []
        for (Md, cd) in by_sector[s]:
            key = (tuple(sorted((i, tuple(sorted(rr.items())))
                                for i, rr in Md.items())),
                   tuple(sorted(cd.items())))
            if key not in seen:
                seen.add(key); out.append((Md, cd))
        by_sector[s] = out
        total += len(out)

    print(f"real: {stats['real']} (fail {stats['real_fail']}); placeholders: "
          f"{stats['ph']} matched {stats['ph_matched']} "
          f"(targeted {stats['ph_targeted']}), unmatched {len(stats['unmatched'])}")
    for s, ig in stats['unmatched'][:20]:
        print(f"    UNMATCHED sector {s} ing_on_dens={ig}")
    print(f"numeric-gate rejections: {stats['numeric_fail']}")
    print(f"store: {len(by_sector)} sectors, {total} transforms")
    out = {"by_sector": by_sector, "gates": stats,
           "prod_point": (prime, kin_values), "N_IND": eng.N_IND,
           "N_DEN": eng.N_DEN, "topology": os.path.basename(topo_dir)}
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"saved -> {out_pkl}")
    ok = stats['real_fail'] == 0 and not stats['unmatched']
    print("ALL PASS" if ok else "GATE INCOMPLETE")
    return out


if __name__ == "__main__":
    topo_dir, symdir, out_pkl = sys.argv[1], sys.argv[2], sys.argv[3]
    prime = int(sys.argv[4]) if len(sys.argv) > 4 else 1009
    build_store(topo_dir, symdir, out_pkl, prime)
