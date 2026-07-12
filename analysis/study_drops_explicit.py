#!/usr/bin/env python
"""Study 10 dropping integrals (P_S.I = lower only) in EXTREMELY EXPLICIT DETAIL.

For each example:
  - the integral, its denominators as explicit momenta, its numerator monomial
  - the pure-symmetry stabilizer of its denominator structure (corner orbit, Schreier)
  - every stabilizer generator that MOVES the numerator: its explicit affine action on
    the numerator slots (small-integer coefficients) and the full value relation
  - where the generator is a base record: the loop-momentum substitution strings
  - the same-(r,s) row reduction: does I -> 0 (symmetric part zero)?
  - symmetry_rule ground truth (drop or not, how many lower terms)
  - a VERDICT TABLE: for the same denominators, every ISP numerator monomial of the
    same degree -> drop or survive.  Pattern lives here.
"""
import os, sys, time, pickle, json, itertools
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
import sympy as sp
from sailir.symmetries import parse_symmetries
import symmetry_engine as SE
from recon_from_swaps import reconstruct
from canonicalize import image_unsigned, sector_of, RED
from symmetry_route import symmetry_rule
import ps_detector as PD

P = 1009; N = 11
def rs(i): return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))
def dstruct(J): return tuple(x if x > 0 else 0 for x in J)
def key(Mc): return PD._mckey(Mc)
def si(x):
    x %= P
    return x - P if x > P // 2 else x
def _props(sec): return [g for g in range(8) if sec >> g & 1]

# ---------- momentum definitions ----------
print("=" * 100)
print("PROPAGATOR / ISP DEFINITIONS (D_i = momentum^2, indices 1-based)")
print("=" * 100)
for i, m in enumerate(SE.PROP_MOM):
    print(f"  D{i+1:<2} = ({m})^2")
print()

# ---------- derive pure-symmetry transforms, keep loop_subst + numeric (M,c) ----------
TA = os.path.join(BASE, "results/kira_reduce_161/sectormappings/TA")
kin = RED.kin; syms = {sp.Symbol(k): v for k, v in kin.items()}
ck = os.path.join(BASE, "analysis", "puresym_rich.pkl")
if os.path.exists(ck):
    RICH = pickle.load(open(ck, "rb"))
else:
    RICH = []
    recs = parse_symmetries(os.path.join(TA, "sectorSymmetries"), N, 2)
    for r in recs:
        ls = list(r.loop_substs)
        if any(rhs == "placeholder" for _, rhs in ls):
            rec = reconstruct(r.ing, _props(r.source_sector))
            if rec is None:
                continue
            ls = rec
        Mf, cf = SE.derive_transform(ls)
        M = {i: {j: int(co.subs(syms)) % P for j, co in Mf[i].items()} for i in Mf}
        c = {i: int(cf[i].subs(syms)) % P for i in cf}
        Msym = {i: {j: str(co) for j, co in Mf[i].items()} for i in Mf}
        csym = {i: str(cf[i]) for i in cf if cf[i] != 0}
        RICH.append((r.source_sector, (M, c), tuple(ls), Msym, csym))
    pickle.dump(RICH, open(ck, "wb"))
print(f"pure symmetry transforms: {len(RICH)}\n", flush=True)
_basekey = {key(mc): (ls, Msym, csym) for _, mc, ls, Msym, csym in RICH}
_by = {}
for S, mc, *_ in RICH:
    _by.setdefault(S, []).append(mc)
_by = list(_by.items())

def sym_transforms(K):
    sK = sector_of(K)
    for S, mcs in _by:
        if (S & sK) == sK:
            for mc in mcs:
                yield mc

# ---------- pure-symmetry corner-orbit stabilizer ----------
_stab = {}
def corner_stab(D):
    if D in _stab:
        return _stab[D]
    orbit = {D: PD._ID}; frontier = [D]; edges = []
    while frontier:
        Kc = frontier.pop()
        for (M, c) in sym_transforms(Kc):
            img = image_unsigned(Kc, M, c)
            if img is None or len(img) != 1:
                continue
            (J, co), = img.items()
            if co % P != 1:
                continue
            edges.append((Kc, (M, c), J))
            if J not in orbit:
                if len(orbit) >= 2000:
                    _stab[D] = ([], 0); return _stab[D]
                orbit[J] = PD.compose(orbit[Kc], (M, c)); frontier.append(J)
    gens = []; seen = set()
    for (Kc, s, J) in edges:
        tji = PD.affine_inv(orbit[J])
        if tji is None:
            continue
        g = PD.compose(PD.compose(orbit[Kc], s), tji); k = key(g)
        if k not in seen:
            seen.add(k); gens.append(g)
    _stab[D] = (gens, len(orbit))
    return _stab[D]

def fmt_row(i, M, c):
    row = M.get(i, {}); parts = []
    for j in sorted(row):
        co = si(row[j])
        parts.append(f"{'+' if co >= 0 else '-'}{abs(co) if abs(co) != 1 else ''}D{j+1}")
    cc = si(c.get(i, 0))
    if cc:
        parts.append(f"{'+' if cc >= 0 else '-'}{abs(cc)}(kin)")
    return f"D{i+1} -> " + (" ".join(parts) if parts else "0")

def fmt_int(I):
    dens = [f"D{i+1}^{I[i]}" if I[i] != 1 else f"D{i+1}" for i in range(N) if I[i] > 0]
    nums = [f"D{i+1}^{-I[i]}" if I[i] != -1 else f"D{i+1}" for i in range(N) if I[i] < 0]
    return f"{list(I)}   dens: {' '.join(dens)}   numerator: {' '.join(nums) if nums else '1'}"

def sym_part(I):
    """Same-(r,s) same-structure closure + RREF; return red({I:1}) (== {} iff sym part 0)."""
    D = dstruct(I); W = rs(I); gens, _ = corner_stab(D)
    seen = {I}; frontier = [I]; raw = []
    while frontier:
        Kn = frontier.pop()
        for g in gens:
            img = image_unsigned(Kn, g[0], g[1])
            if img is None:
                continue
            same = {J: co for J, co in img.items() if rs(J) == W and dstruct(J) == D}
            rel = dict(same); rel[Kn] = (rel.get(Kn, 0) - 1) % P
            rel = {k: v for k, v in rel.items() if v % P}
            if rel:
                raw.append(rel)
            for J in same:
                if J not in seen:
                    seen.add(J); frontier.append(J)
    order = sorted(seen); pos = {v: k for k, v in enumerate(order)}; rules = {}
    def red(d):
        out = {}
        for v, co in d.items():
            t = rules.get(v)
            if t is not None:
                for w, cw in t.items():
                    out[w] = (out.get(w, 0) + co * cw) % P
            else:
                out[v] = (out.get(v, 0) + co) % P
        return {k: x for k, x in out.items() if x % P}
    for rel in raw:
        rr = red(rel)
        if not rr:
            continue
        piv = max(rr, key=lambda v: pos[v]); co = rr.pop(piv); inv = pow(co, P - 2, P)
        newp = {k: (-v * inv) % P for k, v in rr.items()}
        for q in list(rules):
            if piv in rules[q]:
                cc = rules[q].pop(piv)
                for w, cw in newp.items():
                    rules[q][w] = (rules[q].get(w, 0) + cc * cw) % P
                rules[q] = {k: v for k, v in rules[q].items() if v % P}
        rules[piv] = newp
    return red({I: 1}), len(seen)

def drop_truth(I):
    r = symmetry_rule(I)
    return (r is not None) and all(rs(t) < rs(I) for t in r), r

# ---------- pick 10 dropping examples, s in {1,2}, r <= 6, distinct structures ----------
print("scanning for dropping examples ...", flush=True)
cands = []
seen_struct = set()
with open(os.path.join(BASE, "data/pentagonbox_10x_raw_jsonl/multisector_data_worker0.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 37:
            continue
        t = tuple(json.loads(line)["target"])
        W = rs(t)
        if not (1 <= W[1] <= 2 and W[0] <= 6):
            continue
        D = dstruct(t)
        if D in seen_struct:
            continue
        dr, rule = drop_truth(t)
        if dr:
            seen_struct.add(D); cands.append((t, rule))
            print(f"  found #{len(cands)}: {list(t)}  (r,s)={W}", flush=True)
        if len(cands) >= 10:
            break
print(flush=True)

# ---------- the detailed study ----------
for ex, (I, rule) in enumerate(cands, 1):
    W = rs(I); D = dstruct(I)
    print("=" * 100)
    print(f"EXAMPLE {ex}:  {fmt_int(I)}")
    print(f"  (r,s) = {W},  sector mask = {sector_of(I)} (props {[g+1 for g in _props(sector_of(I))]})")
    print("=" * 100)
    gens, orb = corner_stab(D)
    print(f"  corner orbit size {orb}, stabilizer generating set: {len(gens)}")
    movers = []
    for g in gens:
        img = image_unsigned(I, g[0], g[1])
        if img is None:
            continue
        same = {J: co for J, co in img.items() if rs(J) == W and dstruct(J) == D}
        if same != {I: 1}:
            movers.append((g, img, same))
    print(f"  generators that MOVE the numerator: {len(movers)}")
    for mi, (g, img, same) in enumerate(movers[:4], 1):
        M, c = g
        base = _basekey.get(key(g))
        print(f"\n  --- mover {mi} ---")
        if base:
            ls, Msym, csym = base
            print(f"    base record, loop_subst: {ls}")
        else:
            print(f"    composite element (product of base symmetries)")
        for i in range(N):
            if I[i] < 0:
                print(f"    numerator slot: {fmt_row(i, M, c)}")
        # value relation
        same_terms = ", ".join(f"{si(co)}*{list(J)}" for J, co in sorted(same.items()))
        low = {J: co for J, co in img.items() if rs(J) < W}
        print(f"    value relation: I = [same-(r,s)] {same_terms}")
        print(f"                    + [lower] {len(low)} terms"
              + (f", e.g. {si(list(low.values())[0])}*{list(list(low.keys())[0])}" if low else ""))
    surv, dim = sym_part(I)
    print(f"\n  same-(r,s) closure at this structure: {dim} integrals")
    print(f"  SYMMETRIC PART of I: {'ZERO  -> drops' if surv == {} else 'NONZERO -> survives'}")
    if surv:
        st = ", ".join(f"{si(co)}*{list(J)}" for J, co in sorted(surv.items())[:6])
        print(f"    survivor combo: {st}")
    lw = sorted(set(rs(t) for t in rule))
    print(f"  symmetry_rule: {len(rule)} lower terms, weights {lw}")

    # ---------- verdict table: all ISP monomials of the same degree ----------
    s_deg = W[1]
    print(f"\n  VERDICT TABLE  (same denominators, every ISP numerator monomial of degree {s_deg}):")
    isps = [8, 9, 10]
    for combo in itertools.combinations_with_replacement(isps, s_deg):
        t = list(D)
        for j in combo:
            t[j] -= 1
        t = tuple(t)
        mono = "*".join(f"D{j+1}" for j in combo)
        dr, rl = drop_truth(t)
        sv, _ = sym_part(t)
        agree = "" if (sv == {}) == dr else "   <-- sym-part test DISAGREES with truth"
        print(f"    {mono:<12} {str(list(t)):<42} drop={str(dr):<6} sym_part_zero={str(sv == {}):<6}{agree}", flush=True)
    print(flush=True)

print("DONE", flush=True)
