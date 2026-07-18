#!/usr/bin/env python
"""Diagnose the false symmetry-zeros: (1) find the transform(s) whose closure
relation kills corner(1023); (2) verify each suspect transform INDEPENDENTLY by
evaluating the propagators on explicit GF(p)-vector kinematics:

  metric: bilinear form diag(+1,-1,-1,-1,-1,-1) over GF(p), 6 components.
  externals: u1, u2, q constructed exactly with u1.u1=u2.u2=1, u1.u2=y,
  q.q=-1, q.u1=q.u2=0. loops: random GF(p) vectors.

For a map (a, b, swap_u, neg_q) with transform (M, c):
  lhs_g = D_g evaluated at k'_i = sum_j a_ij k_j + b_i q, with externals
          u1' = u2, u2' = u1 (if swap_u), q' = -q (if neg_q)
  rhs_g = sum_j M[g][j] D_j(k) + c_g
Any mismatch on any slot at any random draw = the derived transform is WRONG.
This is engine-independent ground truth (only the metric and the yaml
propagator definitions enter)."""
import os, sys, pickle, random
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import numpy as np
from symmetry_engine_GR import (build_A, make_ctx, transform_of, sp_action,
                                N_IND, N_DEN, PROD)

P, Y = PROD
rng = random.Random(20260717)

# ---- explicit GF(p) kinematic vectors, metric diag(+,-,-,-,-,-) ----
G = np.array([1, -1, -1, -1, -1, -1], dtype=np.int64)


def dot(u, v):
    return int((u * v * G).sum() % P)


def sqrt_mod(a):
    a %= P
    for r in range(P):
        if r * r % P == a:
            return r
    return None


# u1 = (1,0,...); u2 = (y, s, 0, ...) with y^2 - s^2 = 1 -> s^2 = y^2-1
s2 = (Y * Y - 1) % P
s = sqrt_mod(s2)
assert s is not None, "y^2-1 not a QR mod p — pick another component split"
u1 = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
u2 = np.array([Y, s, 0, 0, 0, 0], dtype=np.int64)
# q: q.u1=0 -> q0=0; q.u2=0 -> -s*q1=0 -> q1=0; q.q=-1 -> -(q2^2+...)=-1
q = np.array([0, 0, 1, 0, 0, 0], dtype=np.int64)
assert dot(u1, u1) == 1 and dot(u2, u2) == 1 and dot(u1, u2) == Y % P
assert dot(q, q) == (-1) % P and dot(q, u1) == 0 and dot(q, u2) == 0


def props(k1, k2, k3, qv, u1v, u2v):
    """The 15 GR propagators (integralfamilies.yaml)."""
    def sq(v):
        return dot(v, v)
    return [
        sq(k1), sq(k2), sq(k3), sq((k1 + k2 + k3 - qv) % P),
        (2 * dot((k1 + k2) % P, u1v)) % P,
        (-2 * dot(k1, u2v)) % P,
        (-2 * dot((k1 + k2 + k3) % P, u2v)) % P,
        sq((k1 + k2) % P), sq((k1 + k2 - qv) % P), sq((k2 + k3) % P),
        dot(k2, u2v), dot(k1, u1v), dot(k3, u1v), dot(k1, k3), dot(k1, qv),
    ]


def verify_map(a, b, su, nq, n_trials=5):
    """Numeric ground-truth check of the derived (M, c) on random loop momenta."""
    ctx = make_ctx(P, Y)
    M, c = transform_of(a, b, ctx, su, nq)
    for _ in range(n_trials):
        ks = [np.array([rng.randrange(P) for _ in range(6)], dtype=np.int64)
              for _ in range(3)]
        D = props(*ks, q, u1, u2)
        qp = (-q) % P if nq else q
        u1p, u2p = (u2, u1) if su else (u1, u2)
        kp = [(sum(a[i][j] * ks[j] for j in range(3)) + b[i] * qp) % P
              for i in range(3)]
        Dp = props(*kp, qp, u1p, u2p)
        for g in range(N_IND):
            rhs = (sum(int(M[g, j]) * D[j] for j in range(N_IND)) + int(c[g])) % P
            if Dp[g] % P != rhs:
                return False, g
    return True, None


# ---- part 1: which transforms give corner(1023) a self-relation? ----
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
import canonicalize_GR as CG
corner = tuple([1] * 10 + [0] * 5)
print("transforms applicable to corner(1023) and their corner images:")
suspects = []
store = pickle.load(open(os.path.join(ROOT, "results/gr_transforms.pkl"), "rb"))
for S, mcs in store["by_sector"].items():
    if S != 1023:
        continue
    for (Md, cd) in mcs:
        img = CG.image_unsigned(corner, Md, cd)
        print(f"  src {S}: image {img}")
        if img is not None and img.get(corner, 1) not in (1, None) or \
           (img is not None and len(img) == 1 and corner in img
                and img[corner] != 1):
            suspects.append((Md, cd, img[corner]))
print(f"self-relations with coefficient != 1: {len(suspects)}")

# ---- part 2: numeric verification of EVERY stored 1023 transform ----
# The store keeps only (M,c); to re-verify we need the generating maps. Re-run
# the matching for source sector 1023 records and verify each realized map.
from sailir.symmetries import parse_symmetries
SYMDIR = os.path.join(ROOT, "topology_input/gravity3L/kira_validate/sectormappings/GR")
recs = (parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_IND, n_loops=3)
        + parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_IND, n_loops=3))
import itertools
from symmetry_engine_GR import den_signature, parse_subst, targeted_search, CHECK

mats = [m for m in itertools.product((-1, 0, 1), repeat=9)
        if (lambda x: x[0] * (x[4] * x[8] - x[5] * x[7])
            - x[1] * (x[3] * x[8] - x[5] * x[7] * 0 - x[5] * x[6])
            + x[2] * (x[3] * x[7] - x[4] * x[6]))(m) in (1, -1)]
# NOTE: determinant formula must match the engine exactly — recompute honestly:
mats = [m for m in itertools.product((-1, 0, 1), repeat=9)
        if (m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])) in (1, -1)]
shifts = list(itertools.product((-1, 0, 1), repeat=3))
VARIANTS = [(False, False), (True, False), (False, True), (True, True)]
ctxP, ctxC = make_ctx(*PROD), make_ctx(*CHECK)

n_checked = n_bad = 0
for r in recs:
    if r.source_sector != 1023:
        continue
    present = [g for g in range(N_DEN) if r.source_sector >> g & 1]
    ab = parse_subst(list(r.loop_substs))
    cand = []
    if ab is not None:
        cand = [(ab[0], ab[1], False, False)]
    else:
        for m in mats:
            a = (m[0:3], m[3:6], m[6:9])
            for bsh in shifts:
                for su, nq in VARIANTS:
                    MP, cP = transform_of(a, bsh, ctxP, su, nq)
                    sP = den_signature(MP, cP, P)
                    if all(sP[g] == r.ing[g] for g in present):
                        cand.append((a, bsh, su, nq))
    for (a, bsh, su, nq) in cand:
        ok, slot = verify_map(a, bsh, su, nq)
        n_checked += 1
        tag = "OK " if ok else f"WRONG (slot {slot})"
        print(f"  record src 1023 ing={[r.ing[g] for g in present]} map a={a} "
              f"b={bsh} swap_u={su} neg_q={nq}: {tag}")
        if not ok:
            n_bad += 1
print(f"\nmaps checked: {n_checked}, WRONG: {n_bad}")
