#!/usr/bin/env python
"""Sector 201 (pentagonbox) placeholder ing=(3,0,7,6): legacy recon_from_swaps
finds a map; my targeted solver finds nothing. Print the legacy map and test
which of my constraint rows it violates."""
import os, sys
os.environ['SAILIR_TOPOLOGY'] = 'pentagonbox'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import sympy as sp
from sailir.symmetries import parse_symmetries
from recon_from_swaps import reconstruct
from symmetry_engine import N as NLEG
from symmetry_engine2 import GeneralEngine
from sailir.topology import Topology

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
recs = (parse_symmetries(os.path.join(TA, "sectorSymmetries"), NLEG, 2)
        + parse_symmetries(os.path.join(TA, "sectorRelations"), NLEG, 2))

topo_dir = os.path.join(ROOT, "topology_input/pentagonbox")
topo = Topology.from_dir(topo_dir)
eng = GeneralEngine(topo_dir, 1009, dict(topo.kinematics_values))
ID_VAR = (tuple(range(eng.n_ext)), (1,) * eng.n_ext)
import random
rng = random.Random(0)

for r in recs:
    if r.source_sector not in (201, 210, 228):
        continue
    if not any(rhs == "placeholder" for _, rhs in r.loop_substs):
        continue
    present = [g for g in range(8) if r.source_sector >> g & 1]
    print(f"sector {r.source_sector} ing_on_dens={[r.ing[g] for g in present]}")
    rec = reconstruct(r.ing, present)
    print(f"  legacy recon: {rec}")
    if rec is None:
        continue
    # parse into (a, b) with the general engine's symbol order
    loc = {str(s): s for s in eng.all_syms}
    a = [[0] * eng.n_loop for _ in range(eng.n_loop)]
    b = [[0] * eng.n_ext for _ in range(eng.n_loop)]
    for lhs, rhs in rec:
        i = [str(s) for s in eng.loop_syms].index(lhs)
        e = sp.expand(sp.sympify(rhs, locals=loc))
        for j in range(eng.n_loop):
            a[i][j] = int(e.coeff(eng.loop_syms[j]))
        for ea in range(eng.n_ext):
            b[i][ea] = int(e.coeff(eng.ext_syms[ea]))
    print(f"  as (a, b): a={a} b={b}")
    MP, cP = eng.transform_of(tuple(map(tuple, a)), tuple(map(tuple, b)),
                              eng.ctxP, ID_VAR)
    sig = eng.den_signature(MP, cP, 1009)
    print(f"  general-engine den signature on present: "
          f"{[int(sig[g]) for g in present]} (ing {[r.ing[g] for g in present]})")
    ok = eng.verify_map(tuple(map(tuple, a)), tuple(map(tuple, b)), ID_VAR,
                        eng.ctxP, rng)
    print(f"  numeric verify: {ok}")
    print(f"  shift coefficient range in b: {sorted(set(x for row in b for x in row))}")
