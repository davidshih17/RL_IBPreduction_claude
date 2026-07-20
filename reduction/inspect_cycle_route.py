#!/usr/bin/env python
"""Which of the cycle sectors 83 / 115 is canonical, and where does the
production router send each of the two cycle integrals?"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc

cm = pickle.load(open(_tc.CANON_PKL, "rb"))
canon, rep = set(cm["canonical"]), cm["rep_of"]
A = (3, 2, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, -2, 0)
B = (2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, -1, 0)
for S in (83, 115):
    print(f"sector {S}: canonical={S in canon} rep_of={rep.get(S)}")

import symmetry_route
for name in ("canonical_monolithic_rule", "route", "canonicalize_integral"):
    fn = getattr(symmetry_route, name, None)
    print(f"symmetry_route.{name}: {'present' if fn else 'absent'}")
fn = getattr(symmetry_route, "canonical_monolithic_rule", None)
if fn:
    for t in (A, B):
        try:
            print(f"route({list(t)}) -> {fn(t)}")
        except Exception as e:
            print(f"route({list(t)}) raised {type(e).__name__}: {e}")
