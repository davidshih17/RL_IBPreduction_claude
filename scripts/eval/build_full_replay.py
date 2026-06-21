"""Complete round1+round2+round3 replay: combine all three rounds' substitution
caches, replay the (8,5) start integral through them, and classify the final
expression (masters vs any remaining non-masters). Tells us whether round1+2+3
fully reduces the top sector to masters, or a round4 is still needed.

  cache = round1 replay_state['cache']
          UNION round2 work/results/*.pkl successes
          UNION round3 work/results/*.pkl successes
  final = apply_subs({start: 1}, cache)
"""
import datetime
import glob
import os
import pickle
import sys
import time
from collections import Counter

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_paper_masters_only,
                            set_prime, is_master, weight)

R1 = f'{BASE}/results/pentagonbox_8_5_v6'
R2 = f'{BASE}/results/pentagonbox_8_5_v6_round2'
R3 = f'{BASE}/results/pentagonbox_8_5_v6_round3'
OUT = f'{R3}/full_replay'
PRIME = 1009
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
init_from_topology(Topology.from_dir(f'{BASE}/topology_input/pentagonbox'))
set_prime(PRIME)
set_paper_masters_only(False)          # round3 ran --no-paper-masters-only


def apply_subs(expr, cache, prime, max_iter=2000000):
    changed, it = True, 0
    while changed:
        changed, it = False, it + 1
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
        if it % 50 == 0:
            print(f'    ...iter {it}, |expr|={len(expr)}', flush=True)
        if it > max_iter:
            print(f'  WARN: hit {max_iter} iters', flush=True)
            break
    print(f'  converged in {it} iters', flush=True)
    return expr


def add_results(cache, pattern, label):
    n = 0
    for f in glob.glob(pattern):
        if f.endswith('.bs7.pkl'):
            continue
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        ig = r.get('original_integral')
        if ig is not None and r.get('success'):
            cache[ig] = r.get('final_expr', {})
            n += 1
    print(f'[{time.time()-t0:5.1f}s] +{label} successes: {n}  cache now: {len(cache)}',
          flush=True)


# round1 cache + start
st1 = pickle.load(open(f'{R1}/replay_state.pkl', 'rb'))
start = st1['start_integral']
cache = dict(st1['cache'])
print(f'[{time.time()-t0:5.1f}s] round1 cache: {len(cache)}  start={start}', flush=True)

add_results(cache, f'{R2}/work/results/*.pkl', 'round2')
add_results(cache, f'{R3}/work/results/*.pkl', 'round3')

print(f'[{time.time()-t0:5.1f}s] replaying start I{list(start)} through combined cache...',
      flush=True)
active = apply_subs({start: 1}, cache, PRIME)
print(f'[{time.time()-t0:5.1f}s] final expr terms: {len(active)}', flush=True)

MS = ibp_env.MASTERS_SET
n_paper = n_corner = 0
nonmasters, masters = [], []
for ig in active:
    if ig in MS:
        n_paper += 1
        masters.append(ig)
    elif is_master(ig):
        n_corner += 1
        masters.append(ig)
    else:
        nonmasters.append(ig)

print('\n' + '=' * 60, flush=True)
print('FULL ROUND1+ROUND2+ROUND3 REPLAY RESULT', flush=True)
print('=' * 60, flush=True)
print(f'  start integral : I{list(start)} weight={weight(start)[:2]}', flush=True)
print(f'  total terms    : {len(active)}', flush=True)
print(f'  MASTERS        : {len(masters)}  (paper={n_paper} corner={n_corner})', flush=True)
print(f'  NON-MASTERS    : {len(nonmasters)}', flush=True)
if nonmasters:
    wc = Counter((weight(ig)[0], weight(ig)[1]) for ig in nonmasters)
    print(f'  remaining non-masters by (r,s): {dict(sorted(wc.items()))}', flush=True)
    print('  => NOT fully reduced; a round4 would target these.', flush=True)
else:
    print('  => FULLY REDUCED TO MASTERS. round1+2+3 closes the top sector.', flush=True)

pickle.dump({'start': start, 'prime': PRIME, 'final_expr': active,
             'masters': masters, 'nonmasters': nonmasters, 'n_cache': len(cache),
             'built_at': datetime.datetime.now().isoformat()},
            open(f'{OUT}/full_replay.pkl', 'wb'))
print(f'\n[{time.time()-t0:5.1f}s] wrote {OUT}/full_replay.pkl', flush=True)
