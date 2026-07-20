#!/usr/bin/env python
"""EXPLICIT symmetry analysis of condor job 1863743.1's integral
   T = [1,1,0,1,1,1,1,0,0,1,-1,0,0,0,0]   (sector 635, r=7, s=1)
and its sector corner
   C = [1,1,0,1,1,1,1,0,0,1,0,0,0,0,0]    (the Kira-nosym MASTER).

Reports, with nothing hidden:
 1. FIRE oracle values (if in the table)
 2. basis membership (FIRE-68 / canonical-45 / nosym-155)
 3. sector-level orbit: which sectors are symmetry-related to 635
 4. EVERY applicable transform and its image of T and of C — under the
    production CLEAN-DEN image (flip maps inapplicable) AND under a
    permissive image that shows what sign-carrying (flip) maps would claim
 5. production router verdicts."""
import os, sys, pickle, re, importlib.util
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
os.environ['SAILIR_SECTOR_RANK'] = '1'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
CG = _tc.canonicalize_module()
P, N, N_DEN = CG.P, CG.N, CG.N_DEN

T = (1, 1, 0, 1, 1, 1, 1, 0, 0, 1, -1, 0, 0, 0, 0)
C = (1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0)


def sec(t):
    return sum(1 << i for i in range(N_DEN) if t[i] > 0)


def image_permissive(a, M, c):
    """Like the production image but ALLOWS den rows with coefficient != +1
    (flip maps) — carrying the coefficient as inv(co)^power — to DISPLAY what
    the sign-carrying identities would claim. NOT used in production."""
    base = [0] * N
    pref = 1
    num = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M.get(i)
            if not row or len(row) != 1 or c.get(i, 0) % P:
                return None
            j, co = next(iter(row.items()))
            if j >= N_DEN:
                return None
            base[j] += ai
            if co % P != 1:
                pref = pref * pow(pow(co, P - 2, P), ai, P) % P
        elif ai < 0:
            row = M.get(i, {})
            if not row and not c.get(i, 0):
                return None
            num.append((-ai, row, c.get(i, 0)))
    res = {tuple(base): pref}
    for power, row, const in num:
        for _ in range(power):
            new = {}
            for integ, co in res.items():
                for j, mij in row.items():
                    ni = list(integ); ni[j] -= 1
                    new[tuple(ni)] = (new.get(tuple(ni), 0) + co * mij) % P
                if const:
                    new[integ] = (new.get(integ, 0) + co * const) % P
            res = {k: v for k, v in new.items() if v % P}
    return res


# 1. FIRE values
oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))
for name, X in (("T", T), ("C(corner)", C)):
    sol = oracle["solutions"].get(X)
    print(f"1. FIRE value of {name} {list(X)}: "
          + ("NOT in table" if sol is None else
         ("0 (certified zero)" if not sol else f"{len(sol)} master terms")))

# 2. basis membership
spec = importlib.util.spec_from_file_location(
    "grm", os.path.join(_tc.TOPO_DIR, "GR_masters_dict.py"))
grm = importlib.util.module_from_spec(spec); spec.loader.exec_module(grm)
FIRE = {tuple(m) for s, ms in grm.GR_MASTERS.items() for m in ms}
OURS = {tuple(m) for m in ibp_env.MASTERS_SET}
NOSYM = set()
for ln in open(os.path.join(_tc.TOPO_DIR, "kira_nosym_top/results/GR/masters")):
    m = re.match(r"GR\[([0-9,\-]+)\]", ln.strip())
    if m:
        NOSYM.add(tuple(int(x) for x in m.group(1).split(",")))
for name, X in (("T", T), ("C", C)):
    print(f"2. {name}: FIRE68={X in FIRE} OURS45={X in OURS} NOSYM155={X in NOSYM}")

# 3. sector orbit
cm = pickle.load(open(_tc.CANON_PKL, "rb"))
orbit = sorted(s for s in range(1, 1 << N_DEN) if cm["rep_of"][s] == 635)
print(f"3. sector 635: canonical={635 in set(cm['canonical'])}; "
      f"orbit members (rep 635): {orbit}")

# 4. every applicable transform
from symmetry_route import tkey
for name, X in (("T", T), ("C", C)):
    print(f"4. transforms applicable to {name} {list(X)}:")
    n = 0
    for (M, c) in CG._transforms(X):
        n += 1
        prod = CG.image_unsigned(X, M, c)
        perm = image_permissive(X, M, c)
        if prod is None and perm is None:
            print(f"   #{n}: INAPPLICABLE (den row not single-term/const)")
            continue
        if prod is not None:
            terms = ", ".join(f"{v}*{list(k)}" for k, v in list(prod.items())[:3])
            print(f"   #{n}: CLEAN image ({len(prod)} terms): {terms}"
                  + (" ..." if len(prod) > 3 else ""))
        else:
            terms = ", ".join(f"{v}*{list(k)}" for k, v in list(perm.items())[:3])
            print(f"   #{n}: FLIP-ONLY image (excluded in production; "
                  f"sign-carrying claim, {len(perm)} terms): {terms}"
                  + (" ..." if len(perm) > 3 else ""))
    if n == 0:
        print("   NONE — no symmetry record covers sector 635's cone")

# 5. router verdicts
from symmetry_route import canonical_monolithic_rule
for name, X in (("T", T), ("C", C)):
    r = canonical_monolithic_rule(X)
    print(f"5. router({name}) = "
          + ("SURVIVOR (dispatch worker)" if r is None
             else ("0 (symmetry-zero)" if not r else f"rewrite {len(r)} terms")))
