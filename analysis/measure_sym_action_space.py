#!/usr/bin/env python
"""How much do within-sector symmetry relations ADD to the worker's valid action
space, on TYPICAL expressions of real reductions?

Method: replay the stored paths of sampled sectorrank workers (canonical-sector
targets = the population the future pipeline dispatches). At every step, with the
worker's exact substitution state:
  n_ibp = len(enumerate_valid_actions(target, subs, ...,'subsector'))   [production]
  n_sym = #{ within-sector maps g of the worker's sector :
             image_unsigned(target, g) exists and can ELIMINATE target
             (coefficient of target in the relation != 0, i.e. img[target] != 1) }
(n_sym is the direct-action analog; symmetry has no 'indirect' family.)
Report per-step distributions and the ratio -- is the addition O(1), 10x, ...?
"""
import os, sys, glob, pickle, random
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import (IBPEnvironment, enumerate_valid_actions, get_raw_equation,
                            apply_all_substitutions, is_master)
from canonicalize import image_unsigned, P
from symmetry_route import _within_transforms, _sector_of

env = IBPEnvironment()
rng = random.Random(20260710)

pkls = []
for tag in ("m1_sectorrank", "m2_sectorrank", "m3_sectorrank"):
    fs = sorted(glob.glob(os.path.join(BASE, "results/ab_symmetry", tag,
                                       "design1/work/results/*.pkl")))
    rng.shuffle(fs)
    pkls += [(tag, f) for f in fs[:12]]

MAX_STEPS_PER_WORKER = 60
rows = []          # (tag, sector, |wt|, n_ibp, n_sym)
n_workers = 0

for tag, f in pkls:
    r = pickle.load(open(f, "rb"))
    path = r.get("path") or []
    if not r.get("success") or not path:
        continue
    start = tuple(r["original_integral"])
    S = _sector_of(start)
    wt = _within_transforms(S)
    subs = {}
    ok = True
    steps = path if len(path) <= MAX_STEPS_PER_WORKER else \
        path[:MAX_STEPS_PER_WORKER]
    for (target, ibp_op, delta) in steps:
        target = tuple(target); delta = tuple(delta)
        n_ibp = len(enumerate_valid_actions(target, subs, env.ibp_t, env.li_t,
                                            env.shifts, 'subsector'))
        n_sym = 0
        for (M, c) in wt:
            img = image_unsigned(target, M, c)
            if img is not None and (img.get(target, 0) - 1) % P != 0:
                n_sym += 1
        rows.append((tag, S, len(wt), n_ibp, n_sym))
        # advance the substitution state exactly like the replay
        seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
        raw = get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
        cached = apply_all_substitutions(raw, subs)
        if target not in cached or cached[target] == 0:
            ok = False; break
        co = cached.pop(target); inv = pow(co, P - 2, P)
        subs[target] = {k: (-v * inv) % P for k, v in cached.items() if v % P}
    if ok:
        n_workers += 1

print(f"workers replayed: {n_workers}, steps measured: {len(rows)}", flush=True)
if not rows:
    sys.exit("no data")

ibps = sorted(x[3] for x in rows)
syms = sorted(x[4] for x in rows)
def q(v, p): return v[int(p * (len(v) - 1))]
print(f"\nper-step IBP actions   : median {q(ibps,.5)}  p10 {q(ibps,.1)}  p90 {q(ibps,.9)}  mean {sum(ibps)/len(ibps):.1f}")
print(f"per-step SYM actions   : median {q(syms,.5)}  p10 {q(syms,.1)}  p90 {q(syms,.9)}  mean {sum(syms)/len(syms):.1f}")
zero = sum(1 for x in syms if x == 0)
print(f"steps with ZERO sym actions: {zero}/{len(rows)} ({100*zero/len(rows):.0f}%)")
tot_i = sum(ibps); tot_s = sum(syms)
print(f"aggregate addition: sym/ibp = {tot_s}/{tot_i} = {100*tot_s/max(1,tot_i):.1f}%")

print("\nby sector (|within maps|, steps, mean ibp, mean sym):")
agg = {}
for (tag, S, nwt, ni, ns) in rows:
    a = agg.setdefault(S, [nwt, 0, 0, 0])
    a[1] += 1; a[2] += ni; a[3] += ns
for S in sorted(agg, key=lambda s: -agg[s][1])[:12]:
    nwt, n, si, ss = agg[S]
    print(f"  sector {S:>3}: |wt|={nwt:<3} steps={n:<5} mean_ibp={si/n:6.1f}  mean_sym={ss/n:5.1f}")
