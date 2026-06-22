"""DEV-SIDE bundle builder (uses the SAILIR codebase; runs once on our end).

Produces the two DATA files that the standalone replay package ships with:

  reduction_cache.pkl     -- the combined round1+2+3+4 substitution cache
                             {start_integral, prime, cache}, where `cache` is
                             {integral_tuple -> {integral_tuple: coeff_int}}.
                             Pure builtins only (verified loadable without sailir).

  topology_pentagonbox.json -- the minimal topology constants the master
                             classifier needs: n_indices, n_denominators,
                             isp_positions, family_name, and the explicit
                             master-integral basis (list of tuples).

Source of the cache: results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl,
which already unions the per-round result.pkl caches across rounds 1-4.

Run:
  python replay/build_bundle.py > replay/logs/build_bundle.log 2>&1
"""
import json
import os
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)

from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime

HERE = os.path.dirname(os.path.abspath(__file__))
ALL4 = os.path.join(BASE, 'results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl')
TOPO_DIR = os.path.join(BASE, 'topology_input/pentagonbox')

print(f'loading combined cache from {ALL4}', flush=True)
st = pickle.load(open(ALL4, 'rb'))
cache = st['cache']
start = st['start_integral']
prime = st['prime']
print(f'  start={start}  prime={prime}  cache entries={len(cache)}', flush=True)

# --- 1. write the minimal, clean cache bundle (drop active_expr/log/etc.) ---
out_cache = os.path.join(HERE, 'reduction_cache.pkl')
bundle = {
    'start_integral': tuple(start),
    'prime': int(prime),
    'cache': cache,
    'description': ('Combined SAILIR round1+2+3+4 one-step reduction cache for the '
                    'pentagonbox (8,5) top integral. Replaying {start: 1} through '
                    'this cache to a fixpoint yields the master/corner basis.'),
    'source': 'results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl',
}
with open(out_cache, 'wb') as fp:
    pickle.dump(bundle, fp, protocol=4)   # protocol 4 = portable to py3.4+
print(f'wrote {out_cache}  ({os.path.getsize(out_cache)/1e6:.1f} MB)', flush=True)

# --- 2. extract the topology constants + master basis ---
topo = Topology.from_dir(TOPO_DIR)
init_from_topology(topo)
set_prime(prime)

masters_sorted = sorted(tuple(m) for m in topo.masters)
topo_json = {
    'family_name': topo.family_name,
    'n_indices': topo.n_indices,
    'n_denominators': topo.n_denominators,
    'isp_positions': list(topo.isp_positions),
    'n_masters': len(masters_sorted),
    'masters': [list(m) for m in masters_sorted],
    'note': ('Master basis (Kira `masters` file). is_master(i): i in this set, OR '
             '(i is a corner integral AND i\'s sector is not covered by this set).'),
}
out_json = os.path.join(HERE, 'topology_pentagonbox.json')
with open(out_json, 'w') as fp:
    json.dump(topo_json, fp, indent=1)
print(f'wrote {out_json}', flush=True)
print(f'  family={topo.family_name}  n_indices={topo.n_indices}  '
      f'n_denominators={topo.n_denominators}  isp_positions={topo.isp_positions}  '
      f'n_masters={len(masters_sorted)}', flush=True)
print('BUILD_BUNDLE OK', flush=True)
