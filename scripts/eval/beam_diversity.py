#!/usr/bin/env python3
"""Deeper diversity analysis of a beam checkpoint.

Counts unique beam states at several granularities:
    A. by full expr content
    B. by expr restricted to target sector
    C. by set of non-masters in target sector (just the integral set, ignoring coefficients)
    D. by full subs content
    E. by full path (action history)
    F. by (max_weight, n_non_masters) bucket — coarse "is the search at a similar place"
    G. by sorted multiset of last K actions (K=5 default)

Prints distributions and a clustering view.
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

if len(sys.argv) < 2:
    sys.exit("Usage: beam_diversity.py <path/to/checkpoint> [integral]")
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
print(f'step={ck["step"]}, beam_size={N}, target_sector={target_sector}\n')


def expr_in_sector(s):
    return frozenset(
        (k, v) for k, v in s.expr.items()
        if v != 0 and tuple(get_sector_mask(k)) == target_sector
    )

def nms_set_in_sector(s):
    return frozenset(
        k for k, v in s.expr.items()
        if v != 0 and not is_master(k) and tuple(get_sector_mask(k)) == target_sector
    )

def full_path(s):
    return tuple(s.path)

def subs_frozen(s):
    # Subs is dict-of-dict. Make hashable.
    return frozenset((k, frozenset(v.items())) for k, v in s.subs.items())


def count_unique(states, key_fn, label):
    keys = [key_fn(s) for s in states]
    c = Counter(keys)
    n_unique = len(c)
    print(f'{label:55s} {n_unique:3d} / {len(states)} unique')
    # Show size distribution of duplicate clusters
    sizes = Counter(c.values())
    if n_unique < len(states):
        print(f'  cluster sizes (size -> count): {dict(sorted(sizes.items()))}')


print('=== Diversity metrics ===')
count_unique(beam, lambda s: frozenset(s.expr.items()),     'A. unique by FULL expr content')
count_unique(beam, expr_in_sector,                          'B. unique by expr restricted to target_sector')
count_unique(beam, nms_set_in_sector,                       'C. unique by SET of non-masters in target_sector (no coeffs)')
count_unique(beam, subs_frozen,                             'D. unique by FULL subs content')
count_unique(beam, full_path,                               'E. unique by FULL path (action history)')
count_unique(beam, lambda s: (
    max((weight(k)[0], weight(k)[1]) for k, v in s.expr.items()
        if v != 0 and not is_master(k) and tuple(get_sector_mask(k)) == target_sector),
    sum(1 for k, v in s.expr.items() if v != 0 and not is_master(k)
        and tuple(get_sector_mask(k)) == target_sector)),
                                                            'F. unique by (max_weight, n_non_masters)')
count_unique(beam, lambda s: tuple(sorted(s.path[-5:])),    'G. unique by sorted last-5 actions (multiset)')
count_unique(beam, lambda s: tuple(s.path[-5:]),            'H. unique by ORDERED last-5 actions')
count_unique(beam, lambda s: tuple(sorted(s.path)),         'I. unique by sorted FULL action multiset')

# How many unique scores (up to 3 decimals)?
score_keys = [round(s.score, 3) for s in beam]
unique_scores = len(set(score_keys))
print(f'{"J. unique scores (rounded to 3 decimals)":55s} {unique_scores:3d} / {len(beam)} unique')

print()
print('=== Cluster sizes by expr-content (A) ===')
cl = defaultdict(list)
for i, s in enumerate(beam):
    cl[frozenset(s.expr.items())].append((i, s.score))
for k, members in sorted(cl.items(), key=lambda kv: -len(kv[1])):
    n = len(members)
    if n < 2: continue
    scores = [sc for _, sc in members]
    print(f'  cluster size {n}: state_idxs={[i for i,_ in members]}, scores span {max(scores)-min(scores):.5f}')
