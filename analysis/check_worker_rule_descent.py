#!/usr/bin/env python
"""Check a worker result.pkl: every non-master term of final_expr must sit STRICTLY
below the start integral in the active total order (set SAILIR_SECTOR_RANK to match
the run being checked). Usage: check_worker_rule_descent.py <pkl> <start_csv>"""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonical_rep import tkey
from sailir.ibp_env import is_master

pkl, start_csv = sys.argv[1], sys.argv[2]
start = tuple(int(x) for x in start_csv.split(","))
r = pickle.load(open(pkl, "rb"))
expr = r.get("final_expr", {})
k0 = tkey(start)
above = [t for t in expr if tkey(t) <= k0 and not is_master(tuple(t))]
print(f"result: success={r.get('success')}  steps={r.get('n_steps', r.get('steps'))}  "
      f"terms={len(expr)}")
print(f"non-master terms at-or-above start in the active order: {len(above)}")
for t in above[:8]:
    print(f"   VIOLATION: {list(t)}")
print("DESCENT OK" if not above else "DESCENT FAIL")
