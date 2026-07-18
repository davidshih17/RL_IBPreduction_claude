#!/usr/bin/env python
"""Trajectory-level regression: compare the *_regress arms against *_primefix
at WORKER granularity. Workers are pure functions of (integral, model,
settings) — the orchestrator cache never reaches them (verified: the Condor
worker_args carry only integral/topology/model/beam settings) — so:
  (1) every integral reduced in BOTH runs must have a BIT-IDENTICAL one-step
      result: same final expression, same winning path, same step count;
  (2) the dispatched SETS may differ only by scheduling-race transients
      (dispatch beat a cache hit in one run but not the other) — report them.
"""
import os, glob, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
ARMS = [("gate_small_regress", "gate_small_primefix"),
        ("m1_regress", "m1_primefix"),
        ("m2_regress", "m2_primefix"),
        ("m3_regress", "m3_primefix")]


def load_results(d):
    out = {}
    for f in glob.glob(os.path.join(d, "work/results/*.pkl")):
        try:
            r = pickle.load(open(f, "rb"))
        except Exception:
            continue
        out[tuple(r["original_integral"])] = r
    return out


def canon_expr(e):
    return tuple(sorted((tuple(k), v % 1009) for k, v in e.items() if v % 1009))


def canon_path(p):
    return tuple((tuple(t), op, tuple(d)) for (t, op, d) in (p or []))


all_ok = True
for new, ref in ARMS:
    nd = os.path.join(ROOT, "results/ab_symmetry", new, "design1")
    rd = os.path.join(ROOT, "results/ab_symmetry", ref, "design1")
    if not os.path.exists(os.path.join(nd, "reduction.pkl")):
        print(f"{new}: not finished yet — skipping")
        all_ok = False
        continue
    a, b = load_results(nd), load_results(rd)
    common = set(a) & set(b)
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    n_expr = n_path = n_steps = 0
    for I in common:
        ra, rb = a[I], b[I]
        if canon_expr(ra.get("final_expr", ra.get("expr", {}))) != \
           canon_expr(rb.get("final_expr", rb.get("expr", {}))):
            n_expr += 1
        if canon_path(ra.get("path")) != canon_path(rb.get("path")):
            n_path += 1
        if ra.get("steps") != rb.get("steps"):
            n_steps += 1
    ok = (n_expr == 0 and n_path == 0 and n_steps == 0)
    all_ok &= ok
    print(f"{new} vs {ref}:")
    print(f"  workers: {len(a)} vs {len(b)}; common integrals {len(common)}, "
          f"only-new {len(only_a)}, only-ref {len(only_b)}")
    print(f"  common-integral mismatches: expr {n_expr}, path {n_path}, "
          f"steps {n_steps}  -> {'IDENTICAL' if ok else 'DIVERGENT'}")
    for I in sorted(only_a)[:5]:
        print(f"    only-new  dispatch: {list(I)}")
    for I in sorted(only_b)[:5]:
        print(f"    only-ref  dispatch: {list(I)}")
print("ALL PASS — per-integral trajectories identical"
      if all_ok else "INCOMPLETE / DIVERGENT (see above)")
