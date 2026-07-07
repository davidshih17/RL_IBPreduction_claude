#!/usr/bin/env python
"""Scan all A/B tags under results/ab_symmetry/ and report, for each: done/running,
masters-match gate (baseline==design1, and ==meta_reduce GT), and worker savings."""
import sys, os, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE)
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
ROOT = os.path.join(BASE, "results/ab_symmetry")
meta = pickle.load(open(os.path.join(BASE, "results/meta_reduce/list_TA_reductions.pkl"), "rb"))["reductions"]

def load(tag, mode):
    p = os.path.join(ROOT, tag, mode, "reduction.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

tags = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
print(f"{'tag':<16} {'state':<10} {'base_wk':>7} {'d1_wk':>6} {'routed':>7} {'gate':>18} {'saved':>7}")
print("-" * 82)
n_pass = n_done = 0
for tag in tags:
    b, d = load(tag, "baseline"), load(tag, "design1")
    if b is None or d is None:
        state = f"b={'ok' if b else '..'} d={'ok' if d else '..'}"
        print(f"{tag:<16} {'running':<10} {'':>7} {'':>6} {'':>7} {state:>18}")
        continue
    n_done += 1
    mb = {i: c for i, c in b['final_expr'].items() if c != 0}
    md = {i: c for i, c in d['final_expr'].items() if c != 0}
    ok_bd = (mb == md)                       # gate: baseline (independent IBP) == design1
    gate = "PASS" if ok_bd else "FAIL"
    if gate == "PASS":
        n_pass += 1
    gate_s = f"{gate}(bd={ok_bd})"
    saved = ""
    if b['total_jobs'] > 0:
        saved = f"{100*(b['total_jobs']-d['total_jobs'])/b['total_jobs']:.0f}%"
    print(f"{tag:<16} {'done':<10} {b['total_jobs']:>7} {d['total_jobs']:>6} "
          f"{d.get('n_symmetry_routed',0):>7} {gate_s:>18} {saved:>7}")
print("-" * 82)
print(f"{n_done} done, {n_pass} gate-PASS")
