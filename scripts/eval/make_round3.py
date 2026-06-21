"""Round-3 setup: combine round-1's cache with round-2's results, replay onto
the start integral, and emit the REMAINING non-masters as the round-3 targets.

  combined cache = round1 replay_state['cache']  UNION  round2 work/results/*.pkl
  active_expr    = apply_subs({start: 1}, combined cache)
  round3 targets = { non-master integrals in active_expr }   (is_master, PMO=False)

Writes:
  results/pentagonbox_8_5_v6_round3/replay_state.pkl   (cache + active_expr; the
        orchestrator's --resume-from source)
  results/pentagonbox_8_5_v6_round3/round3_targets.txt (one integral per line)
"""
import datetime
import glob
import os
import pickle
import sys
import time

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_paper_masters_only,
                            set_prime, is_master, is_corner_integral, get_sector)

ROUND1 = f'{BASE}/results/pentagonbox_8_5_v6'
ROUND2 = f'{BASE}/results/pentagonbox_8_5_v6_round2'
OUT = f'{BASE}/results/pentagonbox_8_5_v6_round3'
PRIME = 1009
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
topology = Topology.from_dir(f'{BASE}/topology_input/pentagonbox')
init_from_topology(topology)
set_prime(PRIME)
set_paper_masters_only(False)   # masters = paper masters UNION corner integrals


def apply_subs(expr, cache, prime, max_iter=500000):
    changed = True
    it = 0
    while changed:
        changed = False
        it += 1
        new = {}
        for ig, co in expr.items():
            if co == 0:
                continue
            v = cache.get(ig)
            if v is not None and v != {ig: 1}:
                for k, c in v.items():
                    if c:
                        new[k] = (new.get(k, 0) + co * c) % prime
                changed = True
            else:
                new[ig] = (new.get(ig, 0) + co) % prime
        expr = {k: v for k, v in new.items() if v}
        if it > max_iter:
            print(f'WARN: apply_subs hit {max_iter} iters', flush=True)
            break
    print(f'  apply_subs converged in {it} iters', flush=True)
    return expr


# 1) round-1 cache + start
st1 = pickle.load(open(f'{ROUND1}/replay_state.pkl', 'rb'))
start = st1['start_integral']
cache = dict(st1['cache'])
print(f'[{time.time()-t0:5.1f}s] round1 cache entries: {len(cache)}  start={start}',
      flush=True)

# 2) round-2 results -> cache
n2 = n2skip = 0
for f in glob.glob(f'{ROUND2}/work/results/*.pkl'):
    try:
        r = pickle.load(open(f, 'rb'))
    except Exception:
        n2skip += 1
        continue
    ig = r.get('original_integral')
    if ig is not None and r.get('success'):
        cache[ig] = r.get('final_expr', {})
        n2 += 1
print(f'[{time.time()-t0:5.1f}s] round2 successes added: {n2} '
      f'(skipped {n2skip}); combined cache: {len(cache)}', flush=True)

# 3) replay onto the start integral
print(f'[{time.time()-t0:5.1f}s] replaying start through combined cache...', flush=True)
active = apply_subs({start: 1}, cache, PRIME)
print(f'[{time.time()-t0:5.1f}s] active_expr terms: {len(active)}', flush=True)

# 4) classify -> round-3 targets (non-masters)
MS = ibp_env.MASTERS_SET
paper = corner = []
n_paper = n_corner = 0
nonmasters = []
for ig in active:
    if ig in MS:
        n_paper += 1
    elif is_master(ig):
        n_corner += 1
    else:
        nonmasters.append(ig)
print(f'[{time.time()-t0:5.1f}s] active_expr classification: '
      f'paper-masters={n_paper} corner-masters={n_corner} '
      f'NON-MASTERS(round3 targets)={len(nonmasters)}', flush=True)

from collections import Counter
wc = Counter(sum(get_sector(ig)) for ig in nonmasters)
print(f'  round3 targets by #propagators in sector: {dict(sorted(wc.items()))}',
      flush=True)

# 5) write combined replay_state + target list
pickle.dump({'start_integral': start, 'prime': PRIME, 'cache': cache,
             'active_expr': active,
             'built_at': datetime.datetime.now().isoformat(),
             'n_success': len(cache), 'source_round1': ROUND1,
             'source_round2': ROUND2},
            open(f'{OUT}/replay_state.pkl', 'wb'))
with open(f'{OUT}/round3_targets.txt', 'w') as f:
    for ig in sorted(nonmasters):
        f.write(','.join(str(x) for x in ig) + '\n')
print(f'[{time.time()-t0:5.1f}s] wrote {OUT}/replay_state.pkl', flush=True)
print(f'[{time.time()-t0:5.1f}s] wrote {OUT}/round3_targets.txt '
      f'({len(nonmasters)} targets)', flush=True)
