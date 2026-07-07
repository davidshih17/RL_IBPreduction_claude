#!/usr/bin/env python
"""GATE: canonical_rep MUST equal the validated symmetry_route survivor.

This is the guard against the min-vs-max (anti-survivor) bug: canonical_rep picks a
clean-orbit representative by `_target_key`; the reduction keeps the MAXIMAL one (the
survivor symmetry_route routes everything to). If canonical_rep ever picks the minimum
again (or any member other than the survivor), THIS GATE FAILS.

Checks, over every corner (pure clean orbit) + a sample of numerator integrals:
  (A) canonical_rep(corner) == symmetry_route fixed point (follow symmetry_rule to the
      survivor). This ties the canonicalization to the VALIDATED inference ordering.
  (B) idempotent:       canonical_rep(canonical_rep(I)) == canonical_rep(I)
  (C) orbit-invariant:  every clean-orbit member shares one canonical_rep
  (D) it IS the max:    canonical_rep(I) == max(clean_orbit(I), key=tkey)

Run this before generating any symmetry-enhanced data. Exit code 1 on ANY failure.
"""
import sys, os, itertools
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonical_rep import canonical_rep, clean_orbit, tkey
from symmetry_route import symmetry_rule

NPROP, NIDX = 8, 11
ISP = (8, 9, 10)


def corner(mask):
    return tuple(1 if (mask >> k) & 1 else 0 for k in range(NIDX))


def route_survivor(I):
    """Follow symmetry_rule (the validated routing) to its clean fixed point."""
    seen = set()
    while True:
        r = symmetry_rule(I)
        if r is None:
            return I                       # survivor
        if len(r) != 1:
            return None                    # affine (not a clean 1->1) -> not comparable here
        (J,), = (list(r),)
        if J in seen:
            return None                    # cycle guard (should not happen)
        seen.add(J); I = J


fails = []

# (A) every corner: canonical_rep == symmetry_route survivor
nA = 0
for mask in range(1, 1 << NPROP):
    c = corner(mask)
    cr = canonical_rep(c)
    surv = route_survivor(c)
    if surv is None:
        continue                           # affine-routed corner (shouldn't occur, but skip safely)
    nA += 1
    if tuple(cr) != tuple(surv):
        fails.append(("A survivor", list(c), "canonical_rep=", list(cr), "survivor=", list(surv)))

# build a sample of integrals: corners + 1- and 2-dot / numerator variants
sample = []
for mask in range(1, 1 << NPROP):
    b = list(corner(mask))
    sample.append(tuple(b))
    for s in ISP:                          # one numerator
        t = list(b); t[s] = -1; sample.append(tuple(t))
    p = [k for k in range(NPROP) if b[k]]  # one dot on first present prop
    if p:
        t = list(b); t[p[0]] = 2; sample.append(tuple(t))
sample = list(dict.fromkeys(sample))

# (B) idempotent, (C) orbit-invariant, (D) == max(clean_orbit, tkey)
nB = nC = nD = 0
for I in sample:
    cr = canonical_rep(I)
    nB += 1
    if canonical_rep(cr) != cr:
        fails.append(("B idempotent", list(I), list(cr), list(canonical_rep(cr))))
    orb = clean_orbit(I)
    if orb is not None:
        nD += 1
        mx = max(orb, key=lambda j: (tkey(j), j))
        if tuple(cr) != tuple(mx):
            fails.append(("D is-max", list(I), "cr=", list(cr), "max=", list(mx)))
        nC += 1
        for J in orb:
            if canonical_rep(J) != cr:
                fails.append(("C orbit-invariant", list(I), list(J), list(cr), list(canonical_rep(J))))
                break

print("=" * 72)
print(f"(A) canonical_rep(corner) == symmetry_route survivor : checked {nA} corners")
print(f"(B) idempotent                                       : checked {nB}")
print(f"(C) orbit-invariant                                  : checked {nC}")
print(f"(D) canonical_rep == max(clean_orbit, key=tkey)      : checked {nD}")
print("=" * 72)
if fails:
    print(f"FAIL — {len(fails)} violation(s):")
    for f in fails[:20]:
        print("   ", f)
    sys.exit(1)
print("ALL PASS — canonical_rep is the symmetry_route survivor (max-tkey), everywhere.")
