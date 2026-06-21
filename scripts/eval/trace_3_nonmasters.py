"""Trace the origin of the 3 leftover non-masters from the full replay. For each:
 - does it appear in the orchestrator's own reduction.pkl too?
 - was it one of the 525 round3 targets, or a child discovered mid-run?
 - did a round3 worker run for it, and did that worker SUCCEED or FAIL?
 - what is its value in the combined cache (identity = never reduced)?
"""
import glob
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_paper_masters_only,
                            set_prime, is_master, weight)

R1 = f'{BASE}/results/pentagonbox_8_5_v6'
R2 = f'{BASE}/results/pentagonbox_8_5_v6_round2'
R3 = f'{BASE}/results/pentagonbox_8_5_v6_round3'
init_from_topology(Topology.from_dir(f'{BASE}/topology_input/pentagonbox'))
set_prime(1009)
set_paper_masters_only(False)

# 3 leftover non-masters
fr = pickle.load(open(f'{R3}/full_replay/full_replay.pkl', 'rb'))
three = list(fr['nonmasters'])
print(f'leftover non-masters from full replay: {len(three)}', flush=True)

# orchestrator's own reduction.pkl — does it have non-masters too?
red = pickle.load(open(f'{R3}/reduction.pkl', 'rb'))
print(f'reduction.pkl type={type(red).__name__} '
      f'keys={list(red.keys()) if isinstance(red, dict) else "-"}', flush=True)
red_expr = None
if isinstance(red, dict):
    for k in ('final_expr', 'expr', 'result', 'reduction'):
        if isinstance(red.get(k), dict):
            red_expr = red[k]
            break
if red_expr is not None:
    red_nm = [i for i, c in red_expr.items() if c != 0 and not is_master(i)]
    print(f'reduction.pkl non-masters: {len(red_nm)} -> {[list(i) for i in red_nm]}',
          flush=True)
else:
    print('reduction.pkl: could not locate the expression dict', flush=True)

# round3 targets (the 525)
targets = set()
with open(f'{R3}/round3_targets.txt') as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            targets.add(tuple(int(x) for x in ln.split(',')))
print(f'round3 targets file: {len(targets)}', flush=True)

# index round3 worker results by original_integral -> success
res_by_int = {}
for f in glob.glob(f'{R3}/work/results/*.pkl'):
    if f.endswith('.bs7.pkl'):
        continue
    try:
        r = pickle.load(open(f, 'rb'))
    except Exception:
        continue
    ig = r.get('original_integral')
    if ig is not None:
        res_by_int[ig] = (r.get('success'), f.split('/')[-1])

# combined cache value for each
st1 = pickle.load(open(f'{R1}/replay_state.pkl', 'rb'))
cache = dict(st1['cache'])
for f in glob.glob(f'{R2}/work/results/*.pkl') + glob.glob(f'{R3}/work/results/*.pkl'):
    if f.endswith('.bs7.pkl'):
        continue
    try:
        r = pickle.load(open(f, 'rb'))
    except Exception:
        continue
    ig = r.get('original_integral')
    if ig is not None and r.get('success'):
        cache[ig] = r.get('final_expr', {})

print('\n=== per leftover non-master ===', flush=True)
for ig in three:
    intgt = ig in targets
    ran = res_by_int.get(ig)
    cv = cache.get(ig)
    if cv is None:
        cstat = 'NOT IN CACHE (never submitted/never reduced)'
    elif cv == {ig: 1}:
        cstat = 'cache=IDENTITY (failed/unreduced)'
    else:
        cstat = f'cache=expr({len(cv)} terms)'
    print(f'I{list(ig)} w={weight(ig)[:2]}', flush=True)
    print(f'    in 525 targets: {intgt}', flush=True)
    print(f'    worker ran    : {ran if ran else "NO result.pkl for this integral"}',
          flush=True)
    print(f'    cache status  : {cstat}', flush=True)
