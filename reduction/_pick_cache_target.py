"""Pick a list_TA target slightly below (8,5) to test cache reuse from the (8,5)
reduction. Reports each target's weight (r,s), whether it's top-sector, and
whether it's DIRECTLY a key in the (8,5) reduction cache."""
import pickle
import re
import sys
from pathlib import Path

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
CACHE = sys.argv[1] if len(sys.argv) > 1 else f'{BASE}/results/pentagonbox_8_5_v7_fresh/replay_state.pkl'

st = pickle.load(open(CACHE, 'rb'))
cache_keys = set(st['cache'].keys())
print(f"(8,5) cache: {len(cache_keys)} keys  ({CACHE})")

targets = []
for ln in open(f'{BASE}/from_federica/list_TA'):
    m = re.match(r'TA\[([^\]]+)\]', ln.strip())
    if m:
        targets.append(tuple(int(x) for x in m.group(1).split(',')))
print(f"list_TA: {len(targets)} targets")


def weight(i):
    return (sum(x for x in i if x > 0), -sum(x for x in i if x < 0))


def topsector(i):
    return all(i[p] > 0 for p in range(8))


rows = [(weight(t)[0], weight(t)[1], topsector(t), t in cache_keys, t) for t in targets]
incache = [r for r in rows if r[3]]
print(f"list_TA targets DIRECTLY in the (8,5) cache: {len(incache)} / {len(targets)}")
print(f"(8,5) start weight = (8,5)\n")

# candidates: weight strictly below (8,5), still high (r>=6), sorted heaviest first
cands = sorted([r for r in rows if (r[0], r[1]) < (8, 5) and r[0] >= 6],
               key=lambda r: (-r[0], -r[1]))
print("Candidates (weight < (8,5), r>=6), heaviest first:")
for r, s, ts, inc, t in cands[:25]:
    print(f"  w=({r},{s}) top_sector={int(ts)} in_cache={int(inc)}  TA{list(t)}")
