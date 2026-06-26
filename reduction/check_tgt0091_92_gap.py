#!/usr/bin/env python3
"""Did tgt0091 / tgt0092 close their own chains, or leave the same hole?

For each: is it done? is its final_expr all masters? is the orphan product
2,1,1,0,-2,1,0,1,0,0,0 a key in its cache? does replaying the covered_by_cache
integral through ONLY that target's own reduction.pkl cache fully reduce?
"""
import pickle, sys
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'reduction'))
from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, is_master

PRIME = 1009
init_from_topology(Topology.from_dir(str(BASE / 'topology_input/pentagonbox')))
set_prime(PRIME); set_paper_masters_only(False)

PROD = (2, 1, 1, 0, -2, 1, 0, 1, 0, 0, 0)


def apply_substitutions(expr, cache, prime):
    changed, it = True, 0
    while changed:
        changed, it = False, it + 1
        new = {}
        for integ, coeff in expr.items():
            if coeff == 0:
                continue
            sub = cache.get(integ)
            if sub is not None:
                for si, sc in sub.items():
                    if sc:
                        new[si] = (new.get(si, 0) + coeff * sc) % prime
                changed = True
            else:
                new[integ] = (new.get(integ, 0) + coeff) % prime
        expr = {k: v for k, v in new.items() if v != 0}
        if it > 10000:
            break
    return expr


for tgt, covered in [('tgt0091_w7_3', (1, 1, 1, 0, 1, 1, 0, 1, 0, -3, 0)),
                     ('tgt0092_w7_3', (1, 1, 1, 0, 1, 1, 0, 1, 0, -2, -1))]:
    p = BASE / f'results/meta_reduce/{tgt}/reduction.pkl'
    print(f"\n===== {tgt} =====", flush=True)
    if not p.exists():
        print("  reduction.pkl MISSING (target not finished)", flush=True); continue
    o = pickle.load(open(p, 'rb'))
    cache = o.get('cache', {})
    fe = o.get('final_expr', {})
    fe_nm = [i for i in fe if not is_master(i)]
    print(f"  reduction.pkl: |cache|={len(cache)}  |final_expr|={len(fe)}  "
          f"final_expr non-masters={len(fe_nm)}", flush=True)
    print(f"  start_integral={o.get('start_integral')}", flush=True)
    print(f"  covered integral TA[{','.join(map(str,covered))}] is a cache key: {covered in cache}", flush=True)
    print(f"  orphan product 2,1,1,0,-2,1,0,1,0,0,0 is a cache key: {PROD in cache}", flush=True)
    print(f"  is_master(2,1,1,0,-2,1,0,1,0,0,0) = {is_master(PROD)}", flush=True)
    # replay the covered integral through THIS target's own cache only
    expr = apply_substitutions({covered: 1}, cache, PRIME)
    nm = [i for i in expr if not is_master(i)]
    print(f"  replay TA[{','.join(map(str,covered))}] through {tgt}'s OWN cache -> "
          f"{'FULLY REDUCES' if not nm else str(len(nm)) + ' residual: ' + str(nm[:6])}", flush=True)
    if fe_nm:
        print(f"  !! {tgt} final_expr itself leaves non-masters: {fe_nm[:6]}", flush=True)
