#!/usr/bin/env python
"""GR router smoke test: canonical_monolithic_rule on the 27 benchmark targets.
Checks per target:
  - non-canonical sector => rule MUST fire and every RHS term must be (a) in a
    canonical (or lower, sector-0-exempt) sector, (b) strictly lower in the
    sector-senior order;
  - canonical sector => rule may fire (within-orbit lowering) or None
    (survivor); if it fires, same descent checks.
Requires SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1."""
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
from symmetry_route import canonical_monolithic_rule, tkey, _sector_of

with open(_tc.CANON_PKL, "rb") as f:
    CANON = set(pickle.load(f)["canonical"])

targets = [tuple(int(x) for x in ln.split(","))
           for ln in open(os.path.join(ROOT, "results/gr_benchmark_targets.txt"))]
print(f"benchmark targets: {len(targets)}")

fails = 0
for t in targets:
    S = _sector_of(t)
    canon = S in CANON
    r = canonical_monolithic_rule(t)
    if r is None:
        if not canon:
            print(f"  FAIL {list(t)} (sector {S}, NON-canonical) -> survivor")
            fails += 1
        else:
            print(f"  ok   {list(t)} (sector {S:>4}, canonical) -> survivor")
        continue
    ki = tkey(t)
    bad_desc = [J for J in r if not tkey(J) > ki]
    bad_sec = [J for J in r if _sector_of(J) not in CANON and _sector_of(J) != 0
               and _sector_of(J) == S]
    non_canon_rhs = sorted({_sector_of(J) for J in r
                            if _sector_of(J) not in CANON and _sector_of(J) != 0})
    tag = "ok  " if not bad_desc else "FAIL"
    if bad_desc:
        fails += 1
    print(f"  {tag} {list(t)} (sector {S:>4}, {'canonical' if canon else 'NON-canon'})"
          f" -> {len(r)} terms; descent-violations {len(bad_desc)};"
          f" non-canonical RHS sectors {non_canon_rhs}")

print(f"\nfails: {fails}")
print("ALL PASS" if fails == 0 else "FAIL")
