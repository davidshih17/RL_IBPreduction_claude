#!/usr/bin/env python
"""Per-sector PURE within-sector stabilizer table + coverage census.

For each sector S (1..255): collect every symmetry record that, AT S, acts as a pure
signed slot permutation (present props permute among themselves; every other slot maps
to a single slot with coefficient +-1 and no constant). Close under composition (the
group is tiny). Output:
  - results/pure_within_stabilizers.pkl : {S: [ (perm[11], sign[11]) ... ]} (group,
    identity excluded)
  - coverage census: how many sectors / canonical sectors have a nontrivial group,
    group-size histogram, example orbits of dotted/numerator integrals.
"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonicalize import _SRC

P = 1009; N = 11


def pure_action_at(S, M, c):
    """If (M,c) acts on sector S as a pure signed slot permutation, return
    (perm, sign) with perm[i]=j, sign[i] in {+1,-1}; else None.
    Present props must permute among themselves; ALL slots must map monomially
    (numerator powers can sit on any absent prop or ISP, so every slot matters)."""
    pr = [i for i in range(8) if S >> i & 1]
    perm = [0] * N; sign = [1] * N
    for i in range(N):
        row = M.get(i, {})
        if c.get(i, 0) % P or len(row) != 1:
            return None
        (j, co), = row.items()
        co %= P
        if co == 1:
            sign[i] = 1
        elif co == P - 1:
            sign[i] = -1
        else:
            return None
        perm[i] = j
    img = {perm[i] for i in pr}
    if img != set(pr):
        return None                       # not within-sector at S
    if any(sign[i] != 1 for i in pr):
        return None                       # denominator sign flip = not value-clean here
    if len(set(perm)) != N:
        return None                       # not a bijection on slots
    return (tuple(perm), tuple(sign))


def compose(a, b):
    """Apply a then b: slot i -a-> perm_a[i] -b-> perm_b[perm_a[i]]."""
    pa, sa = a; pb, sb = b
    return (tuple(pb[pa[i]] for i in range(N)),
            tuple(sa[i] * sb[pa[i]] for i in range(N)))


IDENT = (tuple(range(N)), (1,) * N)

table = {}
for S in range(1, 256):
    gens = []
    seen = set()
    for src_sec, mcs in _SRC:
        if (src_sec & S) != S:
            continue
        for (M, c) in mcs:
            a = pure_action_at(S, M, c)
            if a and a != IDENT and a not in seen:
                seen.add(a); gens.append(a)
    # close under composition
    group = {IDENT}
    frontier = [IDENT]
    while frontier:
        g = frontier.pop()
        for s in gens:
            h = compose(g, s)
            if h not in group:
                group.add(h); frontier.append(h)
    group.discard(IDENT)
    if group:
        table[S] = sorted(group)

out = os.path.join(BASE, "results/pure_within_stabilizers.pkl")
with open(out, "wb") as f:
    pickle.dump(table, f)

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))
CANON = set(cm["canonical"])
sizes = {}
for S, g in table.items():
    sizes[len(g) + 1] = sizes.get(len(g) + 1, 0) + 1
print(f"sectors with a NONTRIVIAL pure within-sector group: {len(table)}/255")
print(f"  of the 174 canonical sectors: {sum(1 for S in table if S in CANON)}/174")
print(f"  group-size histogram (incl identity): {dict(sorted(sizes.items()))}")
print(f"saved -> {out}")

# examples: orbit of a dotted/numerator integral in a few sectors
import itertools
shown = 0
for S in sorted(table, key=lambda s: -len(table[s])):
    if S not in CANON:
        continue
    pr = [i for i in range(8) if S >> i & 1]
    a = [0] * N
    for k, i in enumerate(pr):
        a[i] = 1 + (k % 2)                # mixed dots
    numslots = [i for i in range(N) if i not in pr]
    a[numslots[0]] = -1
    a = tuple(a)
    orb = {a}
    for (perm, sign) in table[S]:
        img = [0] * N
        sg = 1
        for i in range(N):
            img[perm[i]] = a[i]
            if a[i] < 0 and sign[i] < 0 and (-a[i]) % 2 == 1:
                sg = -sg
        orb.add(tuple(img))
    print(f"  example sector {S} (|group|={len(table[S])+1}): orbit of {list(a)} has "
          f"{len(orb)} members")
    shown += 1
    if shown >= 4:
        break
