#!/usr/bin/env python
"""Systematic bug audit of the stuck g127 dispatch
   T = (1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0):

 1. sector / canonicality / rank of T
 2. router decision replay (canonical_monolithic_rule) AND an exhaustive
    orbit-minimality check: close T under ALL applicable store transforms and
    verify no strictly-lower image exists that the router missed
 3. the total-order key of T under the worker's exact env (sector-senior),
    sanity comparisons against neighbours
 4. masters recognition under the worker's exact setup (SECTOR_RANK=1,
    canonical masters applied, paper-masters-only): every FIRE master of
    sector 127 AND every master in FIRE's answer for T must satisfy is_master
 5. the actual Condor submit environment of the g127 worker
"""
import os, sys, glob, pickle
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
os.environ['SAILIR_SECTOR_RANK'] = '1'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

T = (1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0)

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)

# ---- 1. sector / canonicality / rank ----
from sector_rank import RANK_IDX
CG = _tc.canonicalize_module()
S = sum(1 << i for i in range(_tc.N_DEN) if T[i] > 0)
canon = pickle.load(open(_tc.CANON_PKL, "rb"))
print(f"1. sector {S}; canonical: {S in set(canon['canonical'])}; "
      f"rep_of[{S}]={canon['rep_of'][S]}; RANK_IDX={RANK_IDX[S]}")

# ---- 2. router decision + exhaustive orbit minimality ----
from symmetry_route import canonical_monolithic_rule, tkey
r = canonical_monolithic_rule(T)
print(f"2. router: {'SURVIVOR' if r is None else f'rewrite {len(r)} terms'}")
seen = {T}
frontier = [T]
best = T
cap_hit = False
while frontier:
    K = frontier.pop()
    for (M, c) in CG._transforms(K):
        img = CG.image_unsigned(K, M, c)
        if img is None:
            continue
        for J in img:
            if J not in seen:
                if len(seen) >= 20000:
                    cap_hit = True
                    break
                seen.add(J)
                frontier.append(J)
                if tkey(J) > tkey(best):
                    best = J
print(f"   orbit closure: {len(seen)} integrals (cap hit: {cap_hit})")
print(f"   lowest-in-order orbit member: {list(best)} "
      f"({'== T (T is orbit minimum, survivor CORRECT)' if best == T else 'LOWER THAN T — ROUTER MISSED A REWRITE'})")

# ---- 3. total-order key under the worker's env ----
import beam_search_v7 as bs7
k = bs7._target_key(T)
print(f"3. worker _target_key(T) = {k}")
print(f"   (rank part = -RANK_IDX[sector] = {-RANK_IDX[S]}; weight r=7 s=2 ok: "
      f"{k[1] == -7 and k[2] == -2})")
# neighbours: a subsector term must compare LOWER (larger key), a same-sector
# lower-weight term must compare lower
sub = (1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0)     # sector 63 (subsector)
low = (1, 1, 1, 1, 1, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0)      # same sector, s=1
print(f"   subsector term lower-in-order: {bs7._target_key(sub) > k}; "
      f"same-sector s=1 lower: {bs7._target_key(low) > k}")

# ---- 4. masters recognition under the worker's setup ----
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
import importlib.util
spec = importlib.util.spec_from_file_location(
    "grm", os.path.join(_tc.TOPO_DIR, "GR_masters_dict.py"))
grm = importlib.util.module_from_spec(spec); spec.loader.exec_module(grm)
m127 = [tuple(m) for m in grm.GR_MASTERS[127]]
bad = [m for m in m127 if not ibp_env.is_master(m)]
print(f"4. sector-127 FIRE masters recognized: {len(m127) - len(bad)}/{len(m127)}")
for m in bad:
    print(f"   NOT-A-MASTER: {list(m)}")
oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))
ans = oracle["solutions"][T]
bad2 = [tuple(m) for m in ans if not ibp_env.is_master(tuple(m))]
print(f"   masters in FIRE's answer for T recognized: "
      f"{len(ans) - len(bad2)}/{len(ans)}")
for m in bad2:
    print(f"   NOT-A-MASTER (in FIRE answer): {list(m)}")

# ---- 5. the actual worker submit environment ----
subs = sorted(glob.glob(os.path.join(ROOT, "results/gr_reduce/g127/work/*.sub")))
for f in subs[:1]:
    env = [l.strip() for l in open(f) if l.startswith("environment")]
    print(f"5. submit env ({os.path.basename(f)}): {env}")
