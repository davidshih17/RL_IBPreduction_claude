"""Merge multiple SAILIR reduction caches into ONE replay_state.pkl, so a single
--resume-from hands an orchestrator all of them at once.

A SOURCE can be:
  - a .pkl (replay_state.pkl / reduction.pkl): uses its ['cache'] dict.
  - a run dir (contains replay_state.pkl): uses that.
  - a results dir (work/results/*.pkl): each per-worker pkl carries
    'original_integral' + 'final_expr' (same fields hierarchical_reduction.py's
    own --resume-from scan reads). Safe on a PARTIAL / still-running dir: any
    mid-write / unreadable pkl is skipped.

Identical integrals reduce identically (deterministic), so merge order is
irrelevant; later sources just .update() over earlier ones.

Usage:
  python merge_caches.py OUT_DIR SRC1 [SRC2 ...]
writes OUT_DIR/replay_state.pkl = {'cache': merged}.
"""
import glob
import os
import pickle
import sys
from pathlib import Path


def load_source(src):
    src = Path(src)
    if src.is_dir():
        rs = src / 'replay_state.pkl'
        if rs.exists():
            return load_source(rs)
        if (src / 'results').is_dir():
            results = src / 'results'
        elif (src / 'work' / 'results').is_dir():   # a run dir
            results = src / 'work' / 'results'
        else:
            results = src
        cache = {}
        for pf in sorted(glob.glob(str(results / '*.pkl'))):
            if pf.endswith('.checkpoint'):
                continue
            try:
                with open(pf, 'rb') as f:
                    r = pickle.load(f)
            except Exception:
                continue
            integ = r.get('original_integral')
            if integ is None:
                continue
            if r.get('success'):
                cache[integ] = r.get('final_expr', r.get('expr', {integ: 1}))
            else:
                cache[integ] = {integ: 1}
        return cache
    obj = pickle.load(open(src, 'rb'))
    return obj['cache'] if (isinstance(obj, dict) and 'cache' in obj) else obj


out_dir = Path(sys.argv[1])
srcs = sys.argv[2:]
assert srcs, "give OUT_DIR then >=1 source"

merged = {}
for s in srcs:
    c = load_source(s)
    new = len(set(c.keys()) - set(merged.keys()))
    print(f"{s}\n  {len(c)} entries  ({new} new)", flush=True)
    merged.update(c)

out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / 'replay_state.pkl'
tmp = out_dir / 'replay_state.pkl.tmp'
with open(tmp, 'wb') as f:
    pickle.dump({'cache': merged}, f)
os.replace(tmp, out)   # atomic: a concurrent reader sees old or new, never partial
print(f"\nMERGED: {len(merged)} entries -> {out}")
