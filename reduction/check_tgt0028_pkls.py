#!/usr/bin/env python3
"""Read-only integrity scan of tgt0028's completed worker pkls before resuming.

Loads the 50 most-recently-written pkls (highest risk of truncation by the power
outage) plus a 100-pkl random sample, and reports how many are unreadable or
malformed vs. how many carry a valid {original_integral, final_expr/success}
record (the structure --resume expects, hierarchical_reduction.py:684-689).
Mirrors the orchestrator's own pickle.load (plain, no custom unpickler)."""
import pickle, glob, os, random

RESULTS = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/meta_reduce/tgt0028_w7_4/work/results"

pkls = [p for p in sorted(glob.glob(os.path.join(RESULTS, "*.pkl")))
        if not p.endswith(".checkpoint")]
print(f"total completed pkls: {len(pkls)}", flush=True)

by_mtime = sorted(pkls, key=os.path.getmtime)
random.seed(0)
sample = list(dict.fromkeys(by_mtime[-50:] + random.sample(pkls, min(100, len(pkls)))))
print(f"scanning {len(sample)} pkls (50 most-recent + random sample)", flush=True)

ok = corrupt = nostruct = 0
good_example = None
for p in sample:
    try:
        with open(p, "rb") as f:
            r = pickle.load(f)
    except Exception as e:
        corrupt += 1
        print(f"  CORRUPT: {os.path.basename(p)} :: {e}", flush=True)
        continue
    if not isinstance(r, dict) or r.get("original_integral") is None:
        nostruct += 1
        keys = list(r.keys()) if isinstance(r, dict) else type(r).__name__
        print(f"  NO-STRUCT: {os.path.basename(p)} :: {keys}", flush=True)
        continue
    ok += 1
    if good_example is None:
        good_example = r

print(f"\nRESULT: sample={len(sample)} ok={ok} corrupt={corrupt} no_struct={nostruct}", flush=True)
if good_example is not None:
    fe = good_example.get("final_expr", good_example.get("expr", {}))
    print(f"example record: keys={list(good_example.keys())}", flush=True)
    print(f"  original_integral={good_example.get('original_integral')} "
          f"success={good_example.get('success')} |final_expr|={len(fe)}", flush=True)
