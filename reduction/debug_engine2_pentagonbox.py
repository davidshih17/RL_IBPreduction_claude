#!/usr/bin/env python
"""Why is legacy-vs-general transform overlap 0/125 on pentagonbox? Take real
(non-placeholder) records, derive (M, c) with BOTH engines from the SAME
loop_subst, and print the first row-level difference."""
import os, sys
os.environ['SAILIR_TOPOLOGY'] = 'pentagonbox'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import numpy as np
import sympy as sp

from sailir.symmetries import parse_symmetries
from symmetry_engine import derive_transform, N as NLEG
from sailir_symmetry import SymmetryReducer
from symmetry_engine2 import GeneralEngine
from sailir.topology import Topology

P = 1009
RED = SymmetryReducer(prime=P)
kinsub = {sp.Symbol(k): v for k, v in RED.kin.items()}

topo_dir = os.path.join(ROOT, "topology_input/pentagonbox")
topo = Topology.from_dir(topo_dir)
eng = GeneralEngine(topo_dir, P, dict(topo.kinematics_values))
ID_VAR = (tuple(range(eng.n_ext)), (1,) * eng.n_ext)

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
recs = (parse_symmetries(os.path.join(TA, "sectorSymmetries"), NLEG, 2)
        + parse_symmetries(os.path.join(TA, "sectorRelations"), NLEG, 2))

shown = 0
for r in recs:
    ls = list(r.loop_substs)
    if any(rhs == "placeholder" for _, rhs in ls):
        continue
    # legacy
    Mf, cf = derive_transform(ls)
    Ml = {i: {j: int(co.subs(kinsub)) % P for j, co in Mf[i].items()}
          for i in Mf}
    Ml = {i: {j: v for j, v in row.items() if v} for i, row in Ml.items()}
    Ml = {i: row for i, row in Ml.items() if row}
    cl = {i: int(v.subs(kinsub)) % P for i, v in cf.items()}
    cl = {i: v for i, v in cl.items() if v}
    # general: parse subst into (a, b)
    nl, ne = eng.n_loop, eng.n_ext
    loc = {str(s): s for s in eng.all_syms}
    a = [[0] * nl for _ in range(nl)]
    b = [[0] * ne for _ in range(nl)]
    for lhs, rhs in ls:
        i = [str(s) for s in eng.loop_syms].index(lhs)
        e = sp.expand(sp.sympify(rhs, locals=loc))
        for j in range(nl):
            a[i][j] = int(e.coeff(eng.loop_syms[j]))
        for ea in range(ne):
            b[i][ea] = int(e.coeff(eng.ext_syms[ea]))
    Mg, cg = eng.transform_of(a, b, eng.ctxP, ID_VAR)
    Mgd = {i: {int(j): int(Mg[i, j] % P) for j in np.nonzero(Mg[i] % P)[0]}
           for i in range(eng.N_IND)}
    Mgd = {i: row for i, row in Mgd.items() if row}
    cgd = {i: int(cg[i] % P) for i in range(eng.N_IND) if cg[i] % P}
    same = (Ml == Mgd and cl == cgd)
    print(f"sector {r.source_sector} subst={ls}: {'SAME' if same else 'DIFFER'}")
    if not same and shown < 2:
        shown += 1
        for i in range(eng.N_IND):
            rl, rg = Ml.get(i, {}), Mgd.get(i, {})
            if rl != rg or cl.get(i, 0) != cgd.get(i, 0):
                print(f"   row {i}: legacy {rl} +{cl.get(i,0)}  vs  "
                      f"general {rg} +{cgd.get(i,0)}")
    if shown >= 2:
        break
