#!/usr/bin/env python
"""Final gates for m1_sectorrank_v2 / m3_sectorrank_v2:
  - every final master is a genuine Kira paper master (post-translation);
  - m3: masters identical to the stored baseline (m3_5prop_deg3/baseline);
  - m1: no stored baseline final exists (baseline never completed either) ->
    membership + count checks only, stated explicitly."""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
PAPER = set(ibp_env.MASTERS_SET)      # original Kira basis (NO canonical swap here)


def masters(path):
    if not os.path.exists(path):
        return None
    r = pickle.load(open(path, "rb"))
    fe = r.get("final_expression", r.get("final_expr", {}))
    return {tuple(k): v for k, v in fe.items()}


for tag, basepath in (("m1_sectorrank_v2", "m1_6prop/baseline/reduction.pkl"),
                      ("m3_sectorrank_v2", "m3_5prop_deg3/baseline/reduction.pkl")):
    new = masters(os.path.join(BASE, f"results/ab_symmetry/{tag}/design1/reduction.pkl"))
    print(f"== {tag} ==")
    nonpaper = [list(k) for k in new if k not in PAPER]
    print(f"   final masters: {len(new)}; NOT in Kira paper basis: {len(nonpaper)} {nonpaper}")
    bas = masters(os.path.join(BASE, "results/ab_symmetry", basepath))
    if bas is None:
        print(f"   baseline final: MISSING ({basepath}) — no equality gate possible")
    else:
        print(f"   masters vs baseline: {'IDENTICAL' if new == bas else 'MISMATCH'} "
              f"({len(new)} vs {len(bas)})")
        if new != bas:
            for k in sorted(set(new) | set(bas)):
                if new.get(k) != bas.get(k):
                    print(f"      {list(k)}: new={new.get(k)} base={bas.get(k)}")
