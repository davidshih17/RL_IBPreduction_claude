#!/usr/bin/env python
"""Cross-check the GR router's symmetry-zero verdicts against the FIRE oracle:
for every benchmark target the router sends to 0, FIRE's exact reduction must
also be 0 (empty solution). Any disagreement = our transform store is unsound
(sign handling / external involutions) — hard stop."""
import os, sys, pickle
assert os.environ.get('SAILIR_TOPOLOGY') == 'gravity3L'
assert os.environ.get('SAILIR_SECTOR_RANK') == '1'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
from symmetry_route import canonical_monolithic_rule

oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))
targets = [tuple(int(x) for x in ln.split(","))
           for ln in open(os.path.join(ROOT, "results/gr_benchmark_targets.txt"))]

false_zero = 0     # router kills an integral FIRE keeps nonzero -> UNSOUND
benign = 0         # router survivor for a FIRE-zero (scaleless) integral ->
#                    fine: the IBP worker reduces it to 0 (standard policy)
for t in targets:
    r = canonical_monolithic_rule(t)
    ours_zero = (r is not None and len(r) == 0)
    fire = oracle["solutions"].get(t)
    fire_zero = (fire is not None and len(fire) == 0)
    if fire is None:
        print(f"  {list(t)}: NOT in oracle (skip)")
        continue
    if ours_zero and not fire_zero:
        false_zero += 1
        print(f"  FALSE ZERO {list(t)}: router kills it, FIRE has "
              f"{len(fire)} terms")
    elif fire_zero and not ours_zero:
        benign += 1
        print(f"  benign     {list(t)}: FIRE-zero (scaleless), router leaves "
              f"it to the IBP worker")
    else:
        print(f"  agree      {list(t)}: zero={ours_zero}" +
              ("" if ours_zero else f" (FIRE terms: {len(fire)})"))
print(f"\nfalse zeros (fatal): {false_zero}, benign scaleless survivors: {benign}")
print("ALL PASS" if false_zero == 0 else "FAIL — transform store unsound")
