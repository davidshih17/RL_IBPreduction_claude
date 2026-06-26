"""Count how many list_TA targets are DIRECTLY keys in one or more reduction
caches -- i.e. 'immediately handled' (their one-step reduction is already in the
cache, so the orchestrator resolves them by substitution with zero new workers).

Reports per-cache coverage AND the COMBINED (union) coverage, against both the
full list_TA and the ISP-clean subset.

Usage:
  python count_cache_coverage.py CACHE1.pkl [CACHE2.pkl ...]
where each CACHE is a replay_state.pkl or a reduction.pkl (both carry 'cache').
"""
import pickle
import re
import sys
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')


def load_cache_keys(p):
    p = Path(p)
    if p.is_dir():
        # partial cache: each completed worker pickle is keyed by its integral,
        # encoded in the filename  async_<id>_<i0>_..._<i10>.pkl  (last 11 ints).
        keys = set()
        for f in p.glob('async_*.pkl'):
            if f.stat().st_size == 0:
                continue
            ints = tuple(int(x) for x in f.stem.split('_')[-11:])
            if len(ints) == 11:
                keys.add(ints)
        return keys
    obj = pickle.load(open(p, 'rb'))
    if isinstance(obj, dict) and 'cache' in obj:
        cache = obj['cache']
    else:
        cache = obj
    return set(cache.keys())


def load_targets(p):
    out = []
    for ln in open(p):
        m = re.match(r'TA\[([0-9,\-]+)\]', ln.strip())
        if m:
            out.append(tuple(int(x) for x in m.group(1).split(',')))
    return out


def cover(targets, keys):
    n = sum(1 for t in targets if t in keys)
    return n, 100.0 * n / len(targets)


cache_paths = sys.argv[1:]
assert cache_paths, "give at least one cache pickle path"

full = load_targets(BASE / 'from_federica/list_TA')
clean = load_targets(BASE / 'from_federica/list_TA_ispclean_by_weight')
print(f"full list_TA: {len(full)} targets | ISP-clean subset: {len(clean)} targets\n")

union = set()
for p in cache_paths:
    keys = load_cache_keys(p)
    union |= keys
    nf, pf = cover(full, keys)
    nc, pc = cover(clean, keys)
    print(f"{p}")
    print(f"  cache keys  : {len(keys)}")
    print(f"  covers full : {nf}/{len(full)} = {pf:.1f}%")
    print(f"  covers clean: {nc}/{len(clean)} = {pc:.1f}%\n")

nf, pf = cover(full, union)
nc, pc = cover(clean, union)
print(f"COMBINED (union of {len(cache_paths)} cache(s)): {len(union)} keys")
print(f"  covers full : {nf}/{len(full)} = {pf:.1f}%")
print(f"  covers clean: {nc}/{len(clean)} = {pc:.1f}%")
