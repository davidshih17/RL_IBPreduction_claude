#!/usr/bin/env python
"""Smoke test of the sector-senior order wiring (run with SAILIR_SECTOR_RANK=1).

Checks, without launching a search:
  - beam_search_v7._target_key, symmetry_route.tkey, canonical_rep.tkey all
    produce the SAME key on the same integral (the shared order);
  - subsector terms compare BELOW parent-sector terms even at higher (r,s);
  - the m2 leakage example [3,0,0,0,-1,1,0,0,0,0,-1] (sector 33, rep 17) now has
    its canonical-sector image BELOW it (so the router will route it away);
  - is_active treats a subsector integral as passenger once _START_SECTOR is set.
"""
import os, sys
assert os.environ.get("SAILIR_SECTOR_RANK") == "1", "run with SAILIR_SECTOR_RANK=1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
import beam_search_v7 as bs7
import symmetry_route as sr
import canonical_rep as cr

I = (3, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1)      # m2 leakage example, sector 33
k1, k2, k3 = bs7._target_key(I), sr.tkey(I), cr.tkey(I)
print(f"shared key on leakage example: bs7={k1}")
assert k1 == k2 == k3, f"ORDER MISMATCH: {k1} {k2} {k3}"
print("  bs7 == symmetry_route == canonical_rep : OK")

# subsector below parent even at higher (r,s): parent sector {D5,D6}=48, child {D5}=16
parent = (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0)   # (r,s)=(2,0), sector 48
child = (0, 0, 0, 0, 7, 0, 0, 0, -2, 0, 0)   # (r,s)=(7,2), subsector 16
assert bs7._target_key(child) > bs7._target_key(parent), "subsector not below parent"
print("subsector (r=7 dots) sorts BELOW its parent (r=2) : OK")

# the canonical-sector image of the leakage example must now sort below it
img = (3, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1)    # same powers moved to sector 17 pattern
assert bs7._sector_mask(I) == 33 and bs7._sector_mask(img) == 17
assert bs7._target_key(img) > bs7._target_key(I), "canonical image not below original"
print("canonical-sector image sorts BELOW the non-canonical original : OK")

# is_active: subsector passenger once _START_SECTOR is set
bs7._START_SECTOR = 48
assert bs7.is_active(parent, (2, 0))
assert not bs7.is_active(child, (2, 0)), "subsector term wrongly active"
print("is_active: same-sector active, subsector passenger : OK")
print("ALL SMOKE CHECKS PASS")
