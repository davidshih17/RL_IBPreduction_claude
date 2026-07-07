#!/usr/bin/env python
"""Symmetry canonicalization for the training/inference pipeline.

canonical_rep(I) maps an integral to the LOWEST-total-lex-weight member of its
clean-permutation orbit, using the WORKERS' total order `_target_key = (-r,-s,|abs|)`
(smaller key = higher in the ordering = eliminated first). Two integrals related by a
clean loop relabeling (value-equal single integrals) map to the same rep; the affine
relations that produce combinations are NOT used here (they are the action-space
relations, not renames). This is the dedup canonicalization -- 1->1, no P_S expansion.

CRITICAL: the order is `_target_key`, matching beam_search_v7 and symmetry_route, so
the canonical basis is confluent with the reduction. (The old min-integer `rep_of` /
`canon()` / 139-sector treatment has been removed from the codebase — only this 174
`_target_key` clean-orbit treatment, the one the successful symmetry inference uses,
remains.)
"""
import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from canonicalize import _transforms, image_unsigned, P


def tkey(i):
    """Workers' total order = beam_search_v7._target_key = (-r, -s, |abs|).
    SMALLER key == higher in the ordering == lower total-lex-weight rep to pick."""
    return (-sum(x for x in i if x > 0), -sum(-x for x in i if x < 0),
            tuple(abs(x) for x in i))


_memo = {}


def clean_orbit(I, cap=4000):
    """Set of integrals reachable from I by CLEAN (single-integral, value-equal)
    symmetry relabelings. Bounded by `cap`."""
    I = tuple(I)
    seen = {I}; frontier = [I]
    while frontier:
        K = frontier.pop()
        for (M, c) in _transforms(K):
            img = image_unsigned(K, M, c)
            if img is None or len(img) != 1:
                continue                          # not a clean rename -> skip (affine relation)
            (J, co), = img.items()
            if co % P != 1:                       # value relation must be +1 (no spurious sign)
                continue
            if J not in seen:
                if len(seen) >= cap:
                    return None                   # runaway -> bail (treat I as its own rep)
                seen.add(J); frontier.append(J)
    return seen


def canonical_rep(I):
    """Lowest-total-lex-weight (= _target_key-minimal) member of I's clean orbit.
    Idempotent. Falls back to I itself if the orbit is a singleton or exceeds cap."""
    I = tuple(I)
    if I in _memo:
        return _memo[I]
    orb = clean_orbit(I)
    rep = I if orb is None else min(orb, key=tkey)
    _memo[I] = rep
    return rep
