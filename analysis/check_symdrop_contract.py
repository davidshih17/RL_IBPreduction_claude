#!/usr/bin/env python
"""Contract check for a SYM_DROP worker result: every same-(r,s)-as-start non-master
in final_expr must be eliminable by the orchestrator router (canonical_monolithic_rule
returns a rule for it). Usage: check_symdrop_contract.py <pkl> <start_csv>
Run with SAILIR_SECTOR_RANK=1."""
import os, sys, pickle
assert os.environ.get("SAILIR_SECTOR_RANK") == "1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
from sailir.ibp_env import is_master
from symmetry_route import canonical_monolithic_rule

pkl, start_csv = sys.argv[1], sys.argv[2]
start = tuple(int(x) for x in start_csv.split(","))
def rs(i): return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))
W = rs(start)

r = pickle.load(open(pkl, "rb"))
expr = r.get("final_expr", {})
atlevel = [tuple(t) for t in expr if rs(t) == W and not is_master(tuple(t))
           and tuple(t) != start]
print(f"success={r.get('success')} steps={r.get('steps')} final terms={len(expr)}")
print(f"same-(r,s) non-master RHS terms: {len(atlevel)}")
bad = 0
for t in atlevel:
    rule = canonical_monolithic_rule(t)
    ok = rule is not None
    print(f"   {list(t)}  router-eliminable: {ok}")
    if not ok:
        bad += 1
print("CONTRACT OK" if bad == 0 else f"CONTRACT VIOLATED for {bad} terms")
