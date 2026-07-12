#!/usr/bin/env python
"""Could within-sector symmetry identities eliminate ACTIVE integrals inside the
one-step worker by pushing them BELOW THE START INTEGRAL'S TOTAL-LEX THRESHOLD?
(Run with SAILIR_SECTOR_RANK=1 — the production order.)

Criterion (the right one, per dshih): integral K is eliminable iff the within-sector
relation system rewrites K into terms ALL strictly below the START's total-order key
(_target_key/tkey) — this includes same-(r,s) rewrites onto lex-lower-than-start
integrals, not just the (r,s)-drop witnesses.

Population (the beam-wander population, not just the winning path targets):
  (a) every integral of the ACTIVE expression along the replayed winning path;
  (b) integrals introduced by SAMPLED CANDIDATE actions at each step (what other
      beam states would materialize).
"""
import os, sys, glob, pickle, random
assert os.environ.get("SAILIR_SECTOR_RANK") == "1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import (IBPEnvironment, enumerate_valid_actions, get_raw_equation,
                            apply_all_substitutions, is_master)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()                      # production master basis
from symmetry_route import _within_transforms, _sector_of, _build_rules_within, tkey

env = IBPEnvironment()
P = 1009


def rs(i):
    return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))


_elim_memo = {}


def eliminable_below(K, S, start_key):
    """Within-sector RREF rewrites K entirely below start_key (masters allowed)?"""
    mk = (S, K)
    if mk in _elim_memo:
        rhs = _elim_memo[mk]
    else:
        rules = _build_rules_within(K, S, cap=3000)
        rhs = rules.get(K) if rules is not None else None
        _elim_memo[mk] = rhs
    if rhs is None:
        return False
    return all(tkey(t) > start_key or is_master(tuple(t)) for t in rhs)


rng = random.Random(20260711)
pkls = []
for tag in ("m1_sectorrank_v2", "m2_hybrid", "m3_sectorrank_v2"):
    fs = sorted(glob.glob(os.path.join(BASE, "results/ab_symmetry", tag,
                                       "design1/work/results/*.pkl")))
    rng.shuffle(fs)
    pkls += [(tag, f) for f in fs[:8]]

MAX_STEPS = 30; CAND_SAMPLE = 30
na = nah = nc = nch = n_workers = 0
for tag, f in pkls:
    r = pickle.load(open(f, "rb"))
    path = r.get("path") or []
    if not r.get("success") or not path:
        continue
    start = tuple(r["original_integral"])
    S = _sector_of(start)
    if not _within_transforms(S):
        n_workers += 1
        continue                     # no within-sector maps -> nothing to test
    start_key = tkey(start); start_w12 = rs(start)
    subs = {}
    expr = {start: 1}
    ok = True
    for (target, ibp_op, delta) in path[:MAX_STEPS]:
        target = tuple(target); delta = tuple(delta)
        # (a) active expression integrals at this state
        for K in expr:
            if is_master(tuple(K)):
                continue
            na += 1
            if eliminable_below(K, S, start_key):
                nah += 1
        # (b) sampled candidate actions' new integrals (the beam-wander population)
        cands = enumerate_valid_actions(target, subs, env.ibp_t, env.li_t,
                                        env.shifts, 'subsector')
        for (op, dl) in rng.sample(cands, min(CAND_SAMPLE, len(cands))):
            seed = tuple(target[i] + dl[i] for i in range(ibp_env.N_INDICES))
            raw = get_raw_equation(env.ibp_t, env.li_t, op, seed)
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            co = cached.pop(target); inv = pow(co, P - 2, P)
            for J in cached:
                if (_sector_of(J) == S and rs(J) >= start_w12
                        and not is_master(tuple(J)) and J not in expr):
                    nc += 1
                    if eliminable_below(J, S, start_key):
                        nch += 1
        # advance along the winning path (stripped active expression)
        raw = get_raw_equation(env.ibp_t, env.li_t, ibp_op,
                               tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES)))
        cached = apply_all_substitutions(raw, subs)
        if target not in cached or cached[target] == 0:
            ok = False; break
        co = cached.pop(target); inv = pow(co, P - 2, P)
        sol = {k: (-v * inv) % P for k, v in cached.items() if v % P}
        subs[target] = sol
        coeff = expr.pop(target, 0)
        for J, cj in sol.items():
            if _sector_of(J) != S or rs(J) < start_w12:
                continue
            v = (expr.get(J, 0) + coeff * cj) % P
            if v:
                expr[J] = v
            else:
                expr.pop(J, None)
    if ok:
        n_workers += 1

print(f"workers replayed: {n_workers}")
print(f"(a) ACTIVE-expression integrals tested: {na}, eliminable below start "
      f"threshold: {nah} ({100*nah/max(1,na):.1f}%)")
print(f"(b) CANDIDATE-introduced integrals tested: {nc}, eliminable below start "
      f"threshold: {nch} ({100*nch/max(1,nc):.1f}%)")
