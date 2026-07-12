#!/usr/bin/env python
"""How often does the odd-reflection DROP WITNESS (I -> -I + lower) fire on integrals
actually encountered inside one-step worker reductions?

Replay sampled worker paths (same machinery as measure_sym_action_space). At every
step, test the step's target for the single-witness drop: some within-sector map g
with image_unsigned(target, g) = {target: -1} + strictly-lower terms. Also test every
NEW same-sector active integral introduced by the step's substitution (the active
bucket population, not just targets)."""
import os, sys, glob, pickle, random
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import get_raw_equation, apply_all_substitutions, is_master, IBPEnvironment
from canonicalize import image_unsigned, P
from symmetry_route import _within_transforms, _sector_of

env = IBPEnvironment()


def rs(i):
    return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))


_wit_memo = {}


def has_witness(I, wt):
    """Single-witness drop: exists g with same-(r,s) part of image == {I: -1}."""
    if I in _wit_memo:
        return _wit_memo[I]
    W = rs(I); out = False
    for (M, c) in wt:
        img = image_unsigned(I, M, c)
        if img is None:
            continue
        lead = {J: co for J, co in img.items() if rs(J) == W}
        if lead == {I: P - 1}:
            out = True; break
    _wit_memo[I] = out
    return out


rng = random.Random(20260711)
pkls = []
for tag in ("m1_sectorrank_v2", "m2_hybrid", "m3_sectorrank_v2"):
    fs = sorted(glob.glob(os.path.join(BASE, "results/ab_symmetry", tag,
                                       "design1/work/results/*.pkl")))
    rng.shuffle(fs)
    pkls += [(tag, f) for f in fs[:12]]

n_targets = 0; n_target_hits = 0
n_active = 0; n_active_hits = 0
n_workers = 0
for tag, f in pkls:
    r = pickle.load(open(f, "rb"))
    path = r.get("path") or []
    if not r.get("success") or not path:
        continue
    start = tuple(r["original_integral"])
    S = _sector_of(start)
    wt = _within_transforms(S)
    if not wt:
        n_workers += 1
        n_targets += len(path[:60])
        continue                      # no within-sector maps -> no witnesses possible
    subs = {}
    start_w12 = rs(start)[:2]
    seen_active = set()
    ok = True
    for (target, ibp_op, delta) in path[:60]:
        target = tuple(target); delta = tuple(delta)
        n_targets += 1
        if has_witness(target, wt):
            n_target_hits += 1
        seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
        raw = get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
        cached = apply_all_substitutions(raw, subs)
        if target not in cached or cached[target] == 0:
            ok = False; break
        co = cached.pop(target); inv = pow(co, P - 2, P)
        sol = {k: (-v * inv) % P for k, v in cached.items() if v % P}
        subs[target] = sol
        # new same-sector active-bucket integrals introduced by this substitution
        for J in sol:
            if (_sector_of(J) == S and rs(J)[:2] >= start_w12
                    and J not in seen_active and not is_master(tuple(J))):
                seen_active.add(J)
                n_active += 1
                if has_witness(J, wt):
                    n_active_hits += 1
    if ok:
        n_workers += 1

print(f"workers replayed: {n_workers}")
print(f"step TARGETS: {n_targets}, with drop witness: {n_target_hits} "
      f"({100*n_target_hits/max(1,n_targets):.1f}%)")
print(f"new ACTIVE-bucket integrals: {n_active}, with drop witness: {n_active_hits} "
      f"({100*n_active_hits/max(1,n_active):.1f}%)")
