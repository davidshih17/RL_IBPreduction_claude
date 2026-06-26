#!/usr/bin/env python3
"""Build the CLEAN combined cache (4 sanctioned pre-meta runs + meta cascade,
NO junk v6/v3/v1/first/second) and save it as
results/meta_reduce/_clean_cache/replay_state.pkl in the {'cache': ...} format
the orchestrator's --resume-from expects (hierarchical_reduction.py:715).

Same sources as replay_list_TA_clean.py (which validated this cache is v6-free).
Used as --resume-from for the 3 standalone reductions of the covered_by_cache
integrals so they reuse all clean reductions and only compute the missing
products."""
import pickle, glob, os, subprocess, time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
R = BASE / 'results'
OUT = R / 'meta_reduce/_clean_cache'
OUT.mkdir(parents=True, exist_ok=True)


def cache_of(o):
    return o['cache'] if isinstance(o, dict) and 'cache' in o else o


def merge_pkl(combined, path):
    try:
        with open(path, 'rb') as f:
            o = pickle.load(f)
    except Exception as e:
        print(f"  SKIP {path}: {e}", flush=True); return
    c = cache_of(o)
    if isinstance(c, dict):
        combined.update(c)
    del o, c


def merge_wr(combined, results_dir):
    for f in glob.glob(os.path.join(results_dir, 'async_*.pkl')):
        if os.path.getsize(f) == 0:
            continue
        try:
            with open(f, 'rb') as fh:
                r = pickle.load(fh)
        except Exception:
            continue
        integ = r.get('original_integral')
        if integ is not None and r.get('success'):
            combined[integ] = r.get('final_expr', {integ: 1})


t0 = time.time()
combined = {}
# 4 sanctioned pre-meta runs (v7_fresh lineage)
merge_pkl(combined, R / 'pentagonbox_8_5_v7_fresh/replay_state.pkl')
merge_wr(combined, str(R / 'pentagonbox_8_5_v7_fresh_round2/work/results'))
merge_pkl(combined, R / 'pentagonbox_7_5_cache_from_85/replay_state.pkl')
merge_pkl(combined, R / 'pentagonbox_7_6_cache_from_85_75/replay_state.pkl')
# meta cascade (clean): _burst_cache + reduction.pkls newer than it + tgt0028 live
burst = R / 'meta_reduce/_burst_cache/replay_state.pkl'
merge_pkl(combined, burst)
newer = subprocess.run(['find', str(R / 'meta_reduce'), '-name', 'reduction.pkl', '-newer', str(burst)],
                       capture_output=True, text=True).stdout.split()
for p in sorted(newer):
    merge_pkl(combined, p)
merge_wr(combined, str(R / 'meta_reduce/tgt0028_w7_4/work/results'))

print(f"clean combined cache: {len(combined)} entries ({time.time()-t0:.0f}s)", flush=True)
tmp = OUT / 'replay_state.pkl.tmp'
with open(tmp, 'wb') as f:
    pickle.dump({'cache': combined}, f, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(tmp, OUT / 'replay_state.pkl')
print(f"saved -> {OUT / 'replay_state.pkl'}", flush=True)
