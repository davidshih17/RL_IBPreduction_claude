#!/usr/bin/env python
"""EMPIRICAL sign test of the GR symmetry identities against the FIRE table.

For every FIRE table target T in a non-canonical sector, apply each verified
transform with a SINGLE-TERM image co * T'. When T' is also a table target,
FIRE gives independent reductions  I[T] = a . G  and  I[T'] = b . G  over the
68 masters. If the value identity I[T] = co * I[T'] holds, then
    r := a - co * b
must vanish modulo the master-pair identities  I[m_nc] = s_p * I[m_c]
(the 23 orbit pairs, sign s_p unknown = what we are measuring). Pairs are
disjoint, so each nonzero r decomposes per pair:  s_p = r[m_c] / (-r[m_nc])
... precisely: lambda on m_nc and -lambda*s_p on m_c, so
    s_p = -r[m_c] * inv(r[m_nc]).
We tally the implied s_p per pair, SEPARATELY for map classes:
  clean : every den row of the map has coefficient +1 (no eikonal flips)
  flip  : some den row coefficient != +1
Prediction (iε-prescription argument): clean maps are true identities and give
a consistent sign table; flip maps give contradictions."""
import os, sys, pickle
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import canonicalize_GR as CG
P = CG.P

oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))
SOL = oracle["solutions"]
MASTERS = [tuple(m) for m in oracle["masters"]]
MSET = set(MASTERS)
cm = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_GR.pkl"), "rb"))
CANON = set(cm["canonical"])


def sec(t):
    return sum(1 << i for i in range(CG.N_DEN) if t[i] > 0)


# the 23 disjoint master pairs (non-canonical paper master, canonical partner)
# discovered by canonical_masters: recompute from the +1 maps for bookkeeping —
# here we only need PAIR MEMBERSHIP, sign-blind: match masters across partner
# sectors by single-term images (either sign).
pair_of = {}          # master -> (pair_id, role) role 0 = non-canonical side
pairs = []
for m in MASTERS:
    S = sec(m)
    if S in CANON:
        continue
    for (M, c) in CG._transforms(m):
        img = CG.image_unsigned(m, M, c)
        if img is None or len(img) != 1:
            continue
        (j, co), = img.items()
        if j in MSET and sec(j) in CANON:
            if m not in pair_of:
                pid = len(pairs)
                pairs.append((m, j))
                pair_of[m] = (pid, 0)
                pair_of[j] = (pid, 1)
            break
print(f"master pairs identified: {len(pairs)}")

targets = set(oracle["targets"])
votes = {"clean": {}, "flip": {}}
checked = {"clean": 0, "flip": 0}
inconsistent = {"clean": 0, "flip": 0}

for T in oracle["targets"]:
    S = sec(T)
    if S in CANON or S == 0:
        continue
    a = SOL.get(T)
    if a is None:
        continue
    present = [g for g in range(CG.N_DEN) if S >> g & 1]
    for (M, c) in CG._transforms(T):
        img = CG.image_unsigned(T, M, c)
        if img is None or len(img) != 1:
            continue
        (Tp, co), = img.items()
        if Tp not in targets or sec(Tp) not in CANON:
            continue
        b = SOL.get(Tp)
        if b is None:
            continue
        # map class: den rows all +1?
        clean = all(len(M.get(g, {})) == 1
                    and next(iter(M[g].values())) % P == 1
                    and not c.get(g, 0) % P
                    for g in present)
        cls = "clean" if clean else "flip"
        # residual r = a - co*b over the 68 masters
        r = {}
        for G, v in a.items():
            r[tuple(G)] = v % P
        for G, v in b.items():
            G = tuple(G)
            r[G] = (r.get(G, 0) - co * v) % P
        r = {G: v for G, v in r.items() if v % P}
        checked[cls] += 1
        if not r:
            continue                          # exact match, no pair info
        # decompose r over the disjoint pairs
        ok = True
        implied = {}
        seen_pids = set()
        for G in r:
            if G not in pair_of:
                ok = False
                break
            seen_pids.add(pair_of[G][0])
        if ok:
            for pid in seen_pids:
                mnc, mc = pairs[pid]
                rn, rc = r.get(mnc, 0), r.get(mc, 0)
                if rn % P == 0 or rc % P == 0:
                    ok = False
                    break
                implied[pid] = (-rc) * pow(rn, P - 2, P) % P
        if not ok:
            inconsistent[cls] += 1
            continue
        for pid, s in implied.items():
            votes[cls].setdefault(pid, {}).setdefault(s, 0)
            votes[cls][pid][s] += 1

for cls in ("clean", "flip"):
    print(f"\n=== map class: {cls} ===")
    print(f"pair-comparisons checked: {checked[cls]}, "
          f"undecomposable residuals: {inconsistent[cls]}")
    for pid, vd in sorted(votes[cls].items()):
        mnc, mc = pairs[pid]
        pretty = {(1): '+1', (P - 1): '-1'}
        vs = {pretty.get(s, s): n for s, n in vd.items()}
        print(f"  pair {pid} {list(mnc)} <-> {list(mc)}: sign votes {vs}")
