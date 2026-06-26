#!/usr/bin/env python3
"""Definitive check for the 4 residuals from the partial-merge replay.

Builds the COMPLETE cascade cache = union of ALL 539 reduction.pkl 'cache'
dicts + tgt0028's live work/results, then:
  (1) reports whether each residual SUB-integral is a key anywhere in the
      complete cache (was it ever reduced by the cascade?), and
  (2) re-replays the 4 residual list_TA integrals.

If the 3 unexpected residuals now fully reduce -> they were merge gaps (the
cascade DID reduce them; the partial cache just lacked the products). If they
stay residual -> genuine gaps the cascade never reduced (covered_by_cache was
one-step-only, not full-chain).

Memory-bounded: load each reduction.pkl, update the combined dict, free it.
"""
import pickle, sys, glob, os, time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'reduction'))
from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, is_master

PRIME = 1009
init_from_topology(Topology.from_dir(str(BASE / 'topology_input/pentagonbox')))
set_prime(PRIME); set_paper_masters_only(False)
print(f"topology=pentagonbox prime={PRIME} paper_masters_only=False", flush=True)


def cache_of(o):
    return o['cache'] if isinstance(o, dict) and 'cache' in o else o


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


t0 = time.time()
combined = {}
redpkls = sorted(glob.glob(str(BASE / 'results/meta_reduce/tgt*/reduction.pkl')))
print(f"merging ALL {len(redpkls)} reduction.pkls (memory-bounded) ...", flush=True)
for i, p in enumerate(redpkls):
    try:
        with open(p, 'rb') as f:
            o = pickle.load(f)
    except Exception as e:
        print(f"  SKIP {p}: {e}", flush=True); continue
    c = cache_of(o)
    if isinstance(c, dict):
        combined.update(c)
    del o, c
    if (i + 1) % 50 == 0:
        print(f"  ... {i+1}/{len(redpkls)}  |combined|={len(combined)}  ({time.time()-t0:.0f}s)", flush=True)

for f in glob.glob(str(BASE / 'results/meta_reduce/tgt0028_w7_4/work/results/async_*.pkl')):
    if os.path.getsize(f) == 0:
        continue
    try:
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
    except Exception:
        continue
    integ = r.get('original_integral')
    if integ is not None and r.get('success'):
        combined[integ] = r.get('final_expr', {integ: 1})
print(f"\nCOMPLETE CACHE: {len(combined)} entries ({time.time()-t0:.0f}s)\n", flush=True)

residual_subs = [
    (2, 1, 1, 0, -2, 1, 0, 1, 0, 0, 0),
    (1, 1, 1, 0, -1, 1, 0, 2, 0, -2, 0),
    (1, 0, 2, 1, 0, 2, 0, 0, -4, 0, 0),
    (2, 0, 1, 1, 0, 2, 0, 0, -4, 0, 0),
    (1, 0, 1, 2, 0, 2, 0, 0, -4, 0, 0),
]
print("=== are the 3-residuals' SUB-integrals keys in the COMPLETE cache? ===", flush=True)
for s in residual_subs:
    print(f"  {'KEY    ' if s in combined else 'MISSING'}  is_master={is_master(s)}  {','.join(map(str, s))}", flush=True)

resid_targets = [
    ('tgt0028', (1, 1, 1, 1, 1, 1, -1, 1, -2, 0, -1)),
    ('covered', (1, 1, 1, 1, 1, 1, 0, 0, -4, 0, 0)),
    ('covered', (1, 1, 1, 0, 1, 1, 0, 1, 0, -3, 0)),
    ('covered', (1, 1, 1, 0, 1, 1, 0, 1, 0, -2, -1)),
]
print("\n=== re-replay the 4 residual list_TA integrals through COMPLETE cache ===", flush=True)
for tag, t in resid_targets:
    expr = apply_substitutions({t: 1}, combined, PRIME)
    nm = [i for i in expr if not is_master(i)]
    print(f"  [{tag}] TA[{','.join(map(str, t))}] -> "
          f"{'FULLY REDUCES' if not nm else str(len(nm)) + ' residual non-masters'}", flush=True)
    for x in nm[:12]:
        print(f"        {','.join(map(str, x))}", flush=True)
