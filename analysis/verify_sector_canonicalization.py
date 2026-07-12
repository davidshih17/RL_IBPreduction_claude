#!/usr/bin/env python
"""Verify the sector-canonicalization theorem against the IMPLEMENTED transform set.

(1) Build the directed corner graph: nodes = 255 corners, edges = clean transforms
    (denominators -> single props).  For each sector S, BFS a path corner(S) ->
    corner(rep(S)) where rep = canonical_rep (the tkey-max survivor); compose the
    (M,c) maps along the path into ONE map.
(2) Power-independence stress test: for each reachable S, apply the composite to
    random integrals of sector S (random dots 1..4, random numerator powers 1..3 on
    up to 3 numerator slots) and check:
      - every image term has (r,s) <= source (r,s), none above;
      - the leading (same-(r,s)) terms live EXACTLY in sector rep(S);
      - the leading denominator-power multiset equals the source multiset (permuted);
      - applicability never failed (image_unsigned never None).
Report reachable/255 and any violations.
"""
import os, sys, random
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, sector_of
from canonical_rep import canonical_rep
import ps_detector as PD

P = 1009; N = 11
def rs(i): return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))
def corner(mask): return tuple(1 if mask >> i & 1 else 0 for i in range(8)) + (0, 0, 0)
def props(mask): return [i for i in range(8) if mask >> i & 1]

# ---- (1) corner graph + BFS with composite maps ----
edges = {}                       # mask -> list of (target_mask, (M,c))
for m in range(1, 256):
    K = corner(m); out = []
    for (M, c) in _transforms(K):
        img = image_unsigned(K, M, c)
        if img is None or len(img) != 1:
            continue
        (J, co), = img.items()
        if co % P != 1:
            continue
        out.append((sector_of(J), (M, c)))
    edges[m] = out

rep_mask = {}
for m in range(1, 256):
    rep_mask[m] = sector_of(canonical_rep(corner(m)))

composite = {}                   # mask -> single (M,c) mapping corner(m) -> corner(rep)
unreachable = []
for m in range(1, 256):
    tgt = rep_mask[m]
    if m == tgt:
        composite[m] = PD._ID; continue
    # BFS
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
        unreachable.append(m); continue
    # reconstruct path, compose
    path = []; v = tgt
    while seen[v] is not None:
        u, mc = seen[v]; path.append(mc); v = u
    g = PD._ID
    for mc in reversed(path):
        g = PD.compose(g, mc)
    composite[m] = g

print(f"sectors reachable to their canonical rep: {256 - 1 - len(unreachable)}/255")
if unreachable:
    print(f"UNREACHABLE sectors: {unreachable}")
    print("  (theorem says the group element exists; if any listed, the implemented")
    print("   record set lacks a directed path -> would need synthesized inverses)")

# sanity: composite maps corner(m) -> corner(rep) exactly
bad_corner = 0
for m, g in composite.items():
    img = image_unsigned(corner(m), g[0], g[1])
    ok = img is not None and len(img) == 1 and next(iter(img)) == corner(rep_mask[m])
    if not ok:
        bad_corner += 1; print(f"  BAD composite at sector {m}: corner image {img}")
print(f"composite corner check: {len(composite) - bad_corner}/{len(composite)} exact")

# ---- (2) power-independence stress test ----
rng = random.Random(20260710)
n_int = 0; n_fail_apply = 0; n_fail_weight = 0; n_fail_sector = 0; n_fail_multiset = 0
for m, g in composite.items():
    pr = props(m); tgt = rep_mask[m]
    numslots = [i for i in range(N) if i not in pr]     # ISPs + absent props
    for _ in range(5):
        a = [0] * N
        for i in pr:
            a[i] = rng.randint(1, 4)                     # random dots
        for i in rng.sample(numslots, rng.randint(1, 3)):
            a[i] = -rng.randint(1, 3)                    # random numerator powers
        a = tuple(a); W = rs(a); n_int += 1
        img = image_unsigned(a, g[0], g[1])
        if img is None:
            n_fail_apply += 1; print(f"  APPLY FAIL sector {m}: {list(a)}"); continue
        lead = [J for J in img if rs(J) == W]
        if any(rs(J) > W for J in img):
            n_fail_weight += 1; print(f"  WEIGHT RAISE sector {m}: {list(a)}")
        if any(sector_of(J) != tgt for J in lead):
            n_fail_sector += 1
            print(f"  LEAD SECTOR FAIL sector {m}->{tgt}: {list(a)} lead sectors "
                  f"{sorted(set(sector_of(J) for J in lead))}")
        srcmult = sorted(x for x in a if x > 0)
        if any(sorted(x for x in J if x > 0) != srcmult for J in lead):
            n_fail_multiset += 1; print(f"  DEN MULTISET FAIL sector {m}: {list(a)}")
print(f"\nstress test: {n_int} random integrals across {len(composite)} sectors")
print(f"  applicability failures : {n_fail_apply}")
print(f"  weight raised          : {n_fail_weight}")
print(f"  leading sector wrong   : {n_fail_sector}")
print(f"  den multiset changed   : {n_fail_multiset}")
print("\nALL ZERO -> canonicalization is always possible and depends only on the sector.")
