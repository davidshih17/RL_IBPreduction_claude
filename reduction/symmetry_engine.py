#!/usr/bin/env python
"""Symmetry application engine for SAILIR (pentagon-box TA).

Stage 1: derive, for any momentum relabeling (Kira loop_subst), the full linear
transform on propagators  D_i -> sum_j M_ij D_j + c_i  (c_i a kinematic constant),
by momentum algebra + re-expression in the propagator basis. This works for
within- AND cross-sector symmetries uniformly (Kira only writes raw matrices for
within-sector ones, so we validate against those, then trust it for cross-sector).

Self-check: derived within-sector matrices must equal Kira's raw 'symmetries'.
"""
import sympy as sp

# ---- momenta ----
K1, K2, P1, P2, P3, P4 = sp.symbols('K1 K2 P1 P2 P3 P4')
MOM = {'k1': K1, 'k2': K2, 'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4}
s12, s23, s34, s45, s51 = sp.symbols('s12 s23 s34 s45 s51')

# loop scalar-product symbols (the 11 unknowns)
sp_syms = sp.symbols('sp_11 sp_22 sp_12 sp_1p1 sp_1p2 sp_1p3 sp_1p4 '
                     'sp_2p1 sp_2p2 sp_2p3 sp_2p4')
(SP11, SP22, SP12, SP1P1, SP1P2, SP1P3, SP1P4, SP2P1, SP2P2, SP2P3, SP2P4) = sp_syms

# product -> scalar-product value (externals are kinematic constants)
PRODMAP = {
    K1**2: SP11, K2**2: SP22, K1*K2: SP12,
    K1*P1: SP1P1, K1*P2: SP1P2, K1*P3: SP1P3, K1*P4: SP1P4,
    K2*P1: SP2P1, K2*P2: SP2P2, K2*P3: SP2P3, K2*P4: SP2P4,
    P1**2: sp.Integer(0), P2**2: sp.Integer(0), P3**2: sp.Integer(0), P4**2: sp.Integer(0),
    P1*P2: s12/2, P1*P3: (s45 - s12 - s23)/2, P1*P4: (s23 - s51 - s45)/2,
    P2*P3: s23/2, P2*P4: (s51 - s23 - s34)/2, P3*P4: s34/2,
}

def dot(a, b):
    """Scalar product of two momenta (sympy expressions in K1,K2,P1..P4)."""
    return sp.expand(sp.expand(a*b).subs(PRODMAP, simultaneous=True))

# ---- propagator momenta (TA), order = index positions 0..10 ----
PROP_MOM = [
    K1,                       # 0  D1
    K1 + P1,                  # 1  D2
    K1 + P1 + P2,             # 2  D3
    K1 + P1 + P2 + P3,        # 3  D4
    K2,                       # 4  D5
    K2 + P1 + P2 + P3,        # 5  D6
    K2 + P1 + P2 + P3 + P4,   # 6  D7
    K1 - K2,                  # 7  D8
    K1 + P1 + P2 + P3 + P4,   # 8  D9  (ISP)
    K2 + P1,                  # 9  D10 (ISP)
    K2 + P1 + P2,             # 10 D11 (ISP)
]
N = 11
Dsyms = sp.symbols('D0:11')   # propagator symbols D0..D10

def linear_in_sp(expr):
    """Return (coeffs over sp_syms, constant) for an expr linear in sp symbols."""
    poly_const = expr.subs({s: 0 for s in sp_syms})
    coeffs = [sp.expand(expr.coeff(s, 1)) for s in sp_syms]
    return coeffs, sp.expand(poly_const)

# Build A: D_i = sum_sp A[i][sp]*sp + const_i
Amat = sp.zeros(N, len(sp_syms))
Aconst = sp.zeros(N, 1)
for i, m in enumerate(PROP_MOM):
    coeffs, const = linear_in_sp(dot(m, m))
    for j, c in enumerate(coeffs):
        Amat[i, j] = c
    Aconst[i, 0] = const

# Solve sp = A^{-1} (D - const)  -> each sp symbol as linear combo of D + const
Ainv = Amat.inv()
Dvec = sp.Matrix([Dsyms[i] for i in range(N)])
SPvec = sp.expand(Ainv * (Dvec - Aconst))           # length 11, entries linear in D
SP_TO_D = {sp_syms[k]: SPvec[k] for k in range(len(sp_syms))}

def derive_transform(loop_subst):
    """loop_subst: list of (lhs, rhs) strings, e.g. [('k1','k1'),('k2','-p2-p3-k2-p1')].
    Returns (M, c): M[i] = dict {j: coeff}, c[i] = kinematic constant, for
    D_i -> sum_j M[i][j] D_j + c[i]."""
    sub = {}
    for lhs, rhs in loop_subst:
        sub[MOM[lhs]] = sp.sympify(rhs, locals=MOM)
    M, c = {}, {}
    for i, m in enumerate(PROP_MOM):
        m_new = sp.expand(m.subs(sub, simultaneous=True))
        d_sym = dot(m_new, m_new)                    # linear in sp_syms + const
        d_in_D = sp.expand(d_sym.subs(SP_TO_D))      # linear in D + const
        row = {}
        for j in range(N):
            cf = sp.simplify(d_in_D.coeff(Dsyms[j], 1))
            if cf != 0:
                row[j] = cf
        const = sp.simplify(d_in_D.subs({Dsyms[j]: 0 for j in range(N)}))
        M[i] = row
        c[i] = const
    return M, c


from itertools import combinations_with_replacement, product as iproduct

def transform_support(a, M, c_nonzero, cap=200000):
    """Apply a symmetry transform to integral index-tuple `a` and return the SET
    of index tuples in the resulting expansion (coefficients ignored).

    M[i]        = dict {j: coeff} for D_i -> sum_j coeff*D_j (+ const).
    c_nonzero[i]= True if the kinematic constant c_i != 0.

    Denominators (a_i>0) must map to a single propagator (clean permutation of
    the sector's propagators) -> contributes a_i to that position.
    Numerators (a_i<0, power p) expand: L_i^p, each of the p slots picks a D_j
    (from M[i]) or the constant; a chosen D_j lowers that position's index by 1.
    Returns None if a denominator maps to a combination (symmetry inapplicable)."""
    n = len(a)
    base = [0] * n
    num_factors = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M.get(i, {})
            if len(row) != 1:
                return None                       # denominator -> combination: inapplicable
            j = next(iter(row))
            base[j] += ai
        elif ai < 0:
            row = M.get(i, {})
            opts = list(row.keys())
            if not opts and not c_nonzero.get(i, False):
                return None                       # numerator maps to nothing
            num_factors.append((-ai, opts, c_nonzero.get(i, False)))
    # per-numerator-factor: distinct D_j usage multisets of total size <= p
    per_factor = []
    for (p, opts, has_const) in num_factors:
        items = list(opts) + (["C"] if has_const else [])
        usages = set()
        for combo in combinations_with_replacement(items, p):
            usages.add(tuple(combo.count(j) for j in opts))
        per_factor.append((opts, list(usages)))
    support = set()
    total = 1
    for _, us in per_factor:
        total *= len(us)
        if total > cap:
            return "LARGE"                        # too big to enumerate; treat specially
    for assignment in iproduct(*[us for (_, us) in per_factor]):
        idx = list(base)
        for (opts, _), usage in zip(per_factor, assignment):
            for j, u in zip(opts, usage):
                idx[j] -= u
        support.add(tuple(idx))
    return support


def raw_matrix_to_Mc(rows):
    """Parse Kira raw matrix rows (list of 'c0 .. c10 const') into (M, c_nonzero)."""
    M, c_nonzero = {}, {}
    for i, row in enumerate(rows):
        t = row.split()
        coeffs, const = t[:N], (t[N] if len(t) > N else "0")
        M[i] = {j: int(coeffs[j]) for j in range(N) if coeffs[j] != "0"}
        c_nonzero[i] = (const != "0")
    return M, c_nonzero


def derived_to_Mc(loop_subst):
    """Derive (M, c_nonzero) via the momentum-algebra engine (for cross-sector)."""
    M_full, c_full = derive_transform(loop_subst)
    M = {i: {j: 1 for j in M_full[i]} for i in M_full}       # only support matters
    c_nonzero = {i: (c_full[i] != 0) for i in c_full}
    return M, c_nonzero


def ing_to_Mc(ing):
    """Fallback for placeholder records: denominator-only permutation from `ing`
    (no numerator info), so it applies only to integrals zero at the -1 slots."""
    M = {g: {t: 1} for g, t in enumerate(ing) if t != -1}
    c_nonzero = {g: False for g in range(len(ing))}
    return M, c_nonzero


def is_placeholder(loop_subst):
    return any(rhs == "placeholder" for _, rhs in loop_subst)


if __name__ == "__main__":
    import os
    ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
    import sys
    sys.path.insert(0, ROOT)
    from sailir.symmetries import parse_symmetries

    TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
    syms = parse_symmetries(os.path.join(TA, "sectorSymmetries"), N, 2)
    recs53 = [r for r in syms if r.source_sector == 53]

    # raw matrices for sector 53
    blocks, cur = [], []
    for ln in open(os.path.join(TA, "symmetries")):
        s = ln.rstrip("\n")
        if s.strip() == "":
            if cur: blocks.append(cur); cur = []
            continue
        cur.append(s)
    if cur: blocks.append(cur)
    raw53 = [b for b in blocks if int(b[2].split()[1]) == 53]

    print(f"sector 53: {len(recs53)} sectorSymmetries records, {len(raw53)} raw matrix blocks\n")
    for r in recs53:
        M, c = derive_transform(list(r.loop_substs))
        print(f"loop_subst={r.loop_substs}")
        for i in range(N):
            terms = " + ".join(f"{v}*D{j}" for j, v in sorted(M[i].items()))
            cc = f"  (+ {c[i]})" if c[i] != 0 else ""
            print(f"   D{i} -> {terms}{cc}")
        print()
    print("=== raw Kira matrices (sector 53) for comparison (row i = D_i image; last col = const) ===")
    for b in raw53:
        print(f"  block target {b[2].split()[1]}:")
        for i, row in enumerate(b[3:3 + N]):
            print(f"   D{i}: {row}")
        print()
