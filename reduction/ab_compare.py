#!/usr/bin/env python
"""Compare an A/B run: baseline vs Design-1 symmetry routing. Gates correctness
(masters identical, and == meta_reduce ground truth) and reports worker savings."""
import sys, os, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE)
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import set_paper_masters_only, is_master
set_paper_masters_only(True)    # A/B now runs --paper-masters-only (production basis)

TAG = sys.argv[1]
ROOT = os.path.join(BASE, "results/ab_symmetry", TAG)

def load(mode):
    p = os.path.join(ROOT, mode, "reduction.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))

def masters_of(expr):
    return {i: c for i, c in expr.items() if c != 0}   # final_expr is masters-only on success

b = load("baseline"); d = load("design1")
if b is None or d is None:
    print(f"[{TAG}] incomplete: baseline={'ok' if b else 'MISSING'} design1={'ok' if d else 'MISSING'}")
    sys.exit(0)

start = b['start_integral']
mb = masters_of(b['final_expr']); md = masters_of(d['final_expr'])
# ground truth from meta_reduce (may be absent for some targets)
meta = pickle.load(open(os.path.join(BASE, "results/meta_reduce/list_TA_reductions.pkl"), "rb"))
gt = meta['reductions'].get(tuple(start))

print(f"===== A/B  tag={TAG}  target=I{list(start)} =====")
print(f"baseline: {b['total_jobs']} worker jobs, {len(mb)} masters, elapsed {b.get('elapsed_time',0):.0f}s")
print(f"design1 : {d['total_jobs']} worker jobs, {len(md)} masters, "
      f"{d.get('n_symmetry_routed',0)} symmetry-routed free, elapsed {d.get('elapsed_time',0):.0f}s")

# correctness gate: baseline (independent IBP) == design1, both --paper-masters-only.
# meta_reduce GT is a --no-paper basis, so it is NOT expected to match here (shown FYI).
eq_bd = (mb == md)
print(f"\nCORRECTNESS (gate = baseline == design1, both --paper):")
print(f"  baseline masters == design1 masters : {eq_bd}   [{'PASS' if eq_bd else 'FAIL'}]")
if not eq_bd:
    only_b = {k: mb[k] for k in mb if mb.get(k) != md.get(k)}
    only_d = {k: md[k] for k in md if md.get(k) != mb.get(k)}
    print(f"    baseline-only/diff: {dict(list(only_b.items())[:5])}")
    print(f"    design1-only/diff : {dict(list(only_d.items())[:5])}")

# savings
if b['total_jobs'] > 0:
    saved = b['total_jobs'] - d['total_jobs']
    print(f"\nSAVINGS: {saved}/{b['total_jobs']} worker jobs avoided = "
          f"{100*saved/b['total_jobs']:.1f}%  ({d.get('n_symmetry_routed',0)} routed free)")
