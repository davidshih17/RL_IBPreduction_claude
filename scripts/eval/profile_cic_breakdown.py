#!/usr/bin/env python3
"""Quick standalone: load a checkpoint, run ONE compute_indirect_cache on
the first beam state's subs, print the per-phase timing breakdown.

Uses the BEAM_PROFILE_CIC-gated instrumentation in
sailir.ibp_env.compute_indirect_substituted.
"""
import os
import sys
import pickle
import time
from pathlib import Path

os.environ['BEAM_PROFILE_CIC'] = '1'

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

CHECKPOINT = (
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/'
    'results/pentagonbox_8_5_v3/work/results/'
    'straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint'
)
TOPOLOGY = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
)
from beam_search_utils import get_sector_mask

t = Topology.from_dir(TOPOLOGY)
ibp_env.init_from_topology(t)
set_prime(1009)
env = IBPEnvironment()

print(f'Loading checkpoint {CHECKPOINT}')
with open(CHECKPOINT, 'rb') as f:
    ck = pickle.load(f)
print(f'  step={ck["step"]}, beam={len(ck["beam"])}')

s0 = ck['beam'][0]
print(f'  state0: |expr|={len(s0.expr)}, |subs|={len(s0.subs)}, |resolved_subs|={len(s0.resolved_subs)}')

# Mimic the worker's per-target-sector filtering before calling compute_indirect_cache.
INTEGRAL = (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)
target_sector = tuple(get_sector_mask(INTEGRAL))
print(f'  target_sector={target_sector}')

fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)
print(f'  After filter: |fsubs|={len(fsubs)}, |fresolved|={len(fresolved)}')

# Warm up sub_cache once so the second call is the cache-hit case
print('\nFirst call (cold sub_cache):')
ibp_env._CIC_PROFILE = []
t0 = time.time()
ic = env.compute_indirect_cache(fsubs, fresolved)
t_total = time.time() - t0
print(f'  wall = {t_total:.3f}s, |indirect_cache| = {len(ic)}')
p = ibp_env._CIC_PROFILE[-1]
for k, v in p.items():
    if isinstance(v, float):
        print(f'  {k:25s} = {v:.3f}s')
    else:
        print(f'  {k:25s} = {v}')

print('\nSecond call (warm sub_cache — matches in-beam steady state):')
ibp_env._CIC_PROFILE = []
t0 = time.time()
ic = env.compute_indirect_cache(fsubs, fresolved)
t_total = time.time() - t0
print(f'  wall = {t_total:.3f}s, |indirect_cache| = {len(ic)}')
p = ibp_env._CIC_PROFILE[-1]
for k, v in p.items():
    if isinstance(v, float):
        print(f'  {k:25s} = {v:.3f}s ({100*v/p["t_total"]:.1f}%)' if k.startswith('t_') and k != 't_total' else f'  {k:25s} = {v:.3f}s')
    else:
        print(f'  {k:25s} = {v}')
