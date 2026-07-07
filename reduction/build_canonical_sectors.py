#!/usr/bin/env python
"""(i) FULL-lattice canonical sector list for the pentagon-box (TA family).

Every symmetry record's `ing` is a permutation of the SOURCE sector's present
propagators onto the TARGET sector's (verified). A record S->T with permutation
pi applies to EVERY subsector U <= present(S) (the dropped slots are already
zero), mapping U -> pi(U) <= present(T). Propagating every record over every
subsector and taking the transitive closure gives the orbit of ALL 256 sectors.
Representative = orbit-MIN sector id. We also record, per sector, the composite
permutation (sector-present-slots -> rep-present-slots) — that is the denominator
relabel `canonicalize()` will use in (ii).

Saves results/canonical_sectors.pkl and prints verification vs Kira.
"""
import os, sys, pickle
from itertools import combinations
from collections import deque, defaultdict
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
NIDX, NPROP = 11, 8

def present(sec): return [i for i in range(NPROP) if (sec >> i) & 1]
def mask(slots):
    m = 0
    for s in slots: m |= (1 << s)
    return m

# ---- gather edges (a -> b, perm a->b) from both files, propagated to subsectors ----
recs = (parse_symmetries(os.path.join(TA, "sectorSymmetries"), NIDX, 2)
        + parse_symmetries(os.path.join(TA, "sectorRelations"), NIDX, 2))
adj = defaultdict(list)              # sector -> list of (other, perm dict self->other)
autos = defaultdict(list)            # sector -> list of within-sector auto perms (present->present)
for r in recs:
    pS = present(r.source_sector)
    pi = {g: r.ing[g] for g in pS}   # source slot -> target slot (verified bijection on pS)
    for k in range(len(pS) + 1):
        for U in combinations(pS, k):
            a = mask(U)
            img = {g: pi[g] for g in U}
            b = mask(img.values())
            if a == b:
                # the restriction of pi to U maps U onto itself: a within-(SUB)sector
                # automorphism. These come from records whose SOURCE is any sector
                # >= U (not only source==target), so we must harvest them at every
                # subset, else within-sector canonicalization is incomplete.
                if any(img[g] != g for g in U):
                    autos[a].append(dict(img))
                continue
            inv = {img[g]: g for g in U}
            adj[a].append((b, dict(img)))
            adj[b].append((a, inv))

# ---- connected components (orbits) over the 256 sectors ----
seen = {}
orbits = []
for s0 in range(1 << NPROP):
    if s0 in seen: continue
    comp = []
    dq = deque([s0]); seen[s0] = len(orbits)
    while dq:
        x = dq.popleft(); comp.append(x)
        for (y, _p) in adj.get(x, ()):
            if y not in seen:
                seen[y] = len(orbits); dq.append(y)
    orbits.append(sorted(comp))

# ---- per-orbit BFS from rep(=min) to assign composite perm sector->rep ----
to_rep = {}      # sector -> {present-slot: rep-present-slot}
rep_of = {}
for comp in orbits:
    rep = comp[0]
    rep_of_local = rep
    to_rep[rep] = {g: g for g in present(rep)}
    rep_of[rep] = rep
    dq = deque([rep])
    while dq:
        a = dq.popleft()
        for (b, perm_ab) in adj.get(a, ()):     # perm_ab: a-slots -> b-slots
            if b in to_rep: continue
            inv_ab = {perm_ab[g]: g for g in perm_ab}   # b-slots -> a-slots
            to_rep[b] = {x: to_rep[a][inv_ab[x]] for x in inv_ab}
            rep_of[b] = rep
            dq.append(b)

# every sector reachable?
for s in range(1 << NPROP):
    rep_of.setdefault(s, s); to_rep.setdefault(s, {g: g for g in present(s)})

# ---- report ----
nontriv_orbits = [c for c in orbits if len(c) > 1]
eliminated = sorted(s for c in orbits for s in c[1:])
print("=" * 72)
print(f"sectors total            : {1<<NPROP}")
print(f"orbits                   : {len(orbits)}")
print(f"  non-trivial (size>1)   : {len(nontriv_orbits)}")
print(f"canonical (representative) sectors : {len(orbits)}")
print(f"eliminated (non-canonical) sectors : {len(eliminated)}")
print("=" * 72)
print("VERIFY vs the 7 top-sector orbits found earlier:")
for top in (152, 153, 195, 204, 216, 217, 241):
    members = sorted(s for c in orbits for s in c if rep_of[s] == rep_of[top])
    print(f"  rep(top {top}) = {rep_of[top]:3d}  full-lattice orbit (top-level subset shown): "
          f"{[m for m in members if bin(m).count('1') >= 3]}")
print(f"  161 -> rep {rep_of[161]}  (expect 152)   "
      f"composite perm: {to_rep[161]}")
print("=" * 72)
print("Idempotency of the SECTOR map: rep_of[rep_of[s]] == rep_of[s] for all s?",
      all(rep_of[rep_of[s]] == rep_of[s] for s in range(1 << NPROP)))

# ---- cross-check vs Kira nonTrivialSector if present ----
ntf = os.path.join(TA, "nonTrivialSector")
if os.path.exists(ntf):
    kira_nt = set()
    for ln in open(ntf):
        for tok in ln.split():
            if tok.lstrip("-").isdigit():
                kira_nt.add(int(tok))
    elim_nt = [s for s in eliminated if s in kira_nt]
    print(f"\nKira nonTrivialSector entries parsed: {len(kira_nt)}")
    print(f"  eliminated sectors that Kira ALSO lists as nonTrivial: {len(elim_nt)} "
          f"-> {sorted(elim_nt)[:20]}")

pickle.dump({"rep_of": rep_of, "to_rep": to_rep, "autos": dict(autos),
             "orbits": orbits},
            open(os.path.join(ROOT, "results/canonical_sectors.pkl"), "wb"))
print("\nsaved -> results/canonical_sectors.pkl")
print("DONE")
