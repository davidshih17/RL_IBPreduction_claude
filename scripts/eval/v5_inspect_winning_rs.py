"""Inspect the RS of the SUCCESSFUL state(s) in a drained v5 run.
For each RS entry (key → value), classify:
  - 'tier_descent': all value-integral weights are strictly LESS than the key's weight
  - 'lateral': some value-integral has the SAME weight as the key
  - 'ascent': some value-integral has HIGHER weight than the key
  - 'masters_only': value is empty (key reduces to masters)

Also count where the start_int / its w-bucket gets substituted into lower stuff.
"""
import pickle, sys, os
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir.ibp_env import (set_prime, set_paper_masters_only,
                             init_from_topology, weight)
from sailir.topology import Topology

topo_dir = sys.argv[2] if len(sys.argv) > 2 else \
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'
topo = Topology.from_dir(topo_dir)
init_from_topology(topo); set_prime(1009); set_paper_masters_only(False)

# Load the ckpt or result
fn = sys.argv[1]
with open(fn, 'rb') as f:
    d = pickle.load(f)

# Pick out beam (ckpt format) or best_state (result format)
if 'beam' in d:
    states = d['beam']
    drained = [s for s in states if s.get('n_non_masters', s.get('expr')) is not None
               and (s.get('n_non_masters', -1) == 0
                    or (s.get('expr') is not None and len(s.get('expr')) == 0))]
    print(f'step={d.get("step")}; n_beam={len(states)}; '
          f'n_drained={len(drained)}')
    targets = drained if drained else states[:1]  # if none drained, examine state 0
else:
    targets = [d]

start_w12 = d.get('start_w12') or (8, 4)  # fallback
print(f'start_w12={start_w12}\n')

def w2(itgr):
    w = weight(itgr); return (w[0], w[1])

for idx, s in enumerate(targets):
    rs = s['resolved_subs']
    print(f'=== state {idx}  RS entries={len(rs)} ===')
    counts = {'tier_descent': 0, 'lateral': 0, 'ascent': 0, 'masters_only': 0}
    # Per (key_w_bucket, max_value_w_bucket) tabulation
    bucket = {}
    for k, v in rs.items():
        kw = w2(k)
        if not v:
            counts['masters_only'] += 1
            bucket.setdefault((kw, 'masters_only'), 0)
            bucket[(kw, 'masters_only')] += 1
            continue
        vws = [w2(vk) for vk in v.keys()]
        max_vw = max(vws)
        if max_vw < kw:
            counts['tier_descent'] += 1
            bucket.setdefault((kw, 'tier_descent'), 0)
            bucket[(kw, 'tier_descent')] += 1
        elif max_vw == kw:
            counts['lateral'] += 1
            bucket.setdefault((kw, 'lateral'), 0)
            bucket[(kw, 'lateral')] += 1
        else:
            counts['ascent'] += 1
            bucket.setdefault((kw, 'ascent'), 0)
            bucket[(kw, 'ascent')] += 1
    print(f'  counts: {counts}')
    # show per-key-weight bucket distribution sorted by key-weight desc
    by_kw = sorted({k[0] for k in bucket}, reverse=True)
    print(f'  per-key-weight breakdown:')
    for kw in by_kw[:12]:
        td = bucket.get((kw, 'tier_descent'), 0)
        la = bucket.get((kw, 'lateral'), 0)
        ms = bucket.get((kw, 'masters_only'), 0)
        as_ = bucket.get((kw, 'ascent'), 0)
        print(f'    kw={kw}  tier_descent={td}  lateral={la}  '
              f'masters_only={ms}  ascent={as_}')
    print()
