#!/usr/bin/env python3
"""Cycle detection for beam states.

Checks each beam state's action path for evidence of cycling:
    1. Repeated identical (target, ibp_op, delta) tuples in the same state
    2. Histogram of how many times the same target gets reduced repeatedly
    3. Distribution of last 20 distinct targets across beam states
    4. Max-weight progression of the BEST beam state over time (would need
       multiple checkpoints — skipped here)

A beam that's evolving healthily has:
    - Few exact-repeat (target, ibp_op, delta) tuples per state path
    - Wide spread of targets reduced over recent steps
    - Many distinct integrals appearing as targets across the beam

A beam that's cycling has:
    - Same actions repeated many times in a state's path
    - Beam converging on the same handful of targets
"""
import sys
import pickle
from pathlib import Path
from collections import Counter, defaultdict

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime, weight, is_master
from beam_search_utils import get_sector_mask

ckpath = sys.argv[1]
integral_str = sys.argv[2] if len(sys.argv) > 2 else '-1,2,1,0,1,2,1,1,-3,0,0'
TOPO = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

ibp_env.init_from_topology(Topology.from_dir(TOPO))
set_prime(1009)
target_sector = tuple(get_sector_mask(tuple(int(x) for x in integral_str.split(','))))

with open(ckpath, 'rb') as f:
    ck = pickle.load(f)
beam = ck['beam']
N = len(beam)
print(f'step={ck["step"]}, beam_size={N}, best_weight_ever={ck["best_weight_ever"]}')
print(f'steps_since_improvement={ck.get("steps_since_improvement", 0)}\n')

# 1. Exact-repeat actions in each state path
print('=== Per-state action-path analysis ===')
print(f"{'idx':>4s} {'path_len':>9s} {'unique':>7s} {'dup%':>6s} {'max_repeat':>11s} | sample dup")
print('-' * 90)
total_dup_pct = 0
total_max_repeat = 0
for i, s in enumerate(beam):
    path = s.path
    c = Counter(path)
    n_unique = len(c)
    n_total = len(path)
    n_dup = n_total - n_unique
    dup_pct = 100.0 * n_dup / n_total if n_total else 0
    most_common = c.most_common(1)
    max_repeat = most_common[0][1] if most_common else 0
    total_dup_pct += dup_pct
    total_max_repeat = max(total_max_repeat, max_repeat)
    if i < 5 or i >= N-3 or max_repeat > 2:
        sample = most_common[0][0] if most_common and max_repeat > 1 else ''
        sample_str = f'(t=I{list(sample[0])[:6]}..., op={sample[1]})' if sample else ''
        print(f'{i:4d} {n_total:9d} {n_unique:7d} {dup_pct:5.1f}% {max_repeat:11d} | {sample_str}')

print(f'\navg path-dup%: {total_dup_pct/N:.1f}%')
print(f'max single-action-repeat across all states: {total_max_repeat}')

# 2. Target hot-spots: which integrals are being reduced over and over across the beam?
print('\n=== Target hot-spots (recent 30 steps across all beam states) ===')
target_counts = Counter()
for s in beam:
    for t, op, d in s.path[-30:]:
        target_counts[t] += 1
print(f'Number of distinct targets in recent 30-step window across beam: {len(target_counts)}')
print(f'Top 10 most-reduced targets:')
for t, n in target_counts.most_common(10):
    print(f'  I{list(t)} -> reduced {n} times')

# 3. Across-state target diversity at THIS step
print('\n=== Across-beam latest-target spread ===')
latest_targets = Counter()
for s in beam:
    if s.path:
        latest_targets[s.path[-1][0]] += 1
print(f'Distinct latest-target integrals across {N} beam states: {len(latest_targets)}')
for t, n in latest_targets.most_common(5):
    print(f'  I{list(t)} -> {n} beam states targeted it last')

# 4. Action diversity in latest action
latest_actions = Counter()
for s in beam:
    if s.path:
        t, op, d = s.path[-1]
        latest_actions[(op, d)] += 1
print(f'Distinct latest (ibp_op, delta) pairs across beam: {len(latest_actions)}')
print(f'(if all 40 states had the same op+delta, that would mean the dedup is fake diversity)')
