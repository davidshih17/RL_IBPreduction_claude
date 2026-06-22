"""Compare the paper-masters-only final expressions of the two (8,5) reductions.

  A = fresh round2 : results/pentagonbox_8_5_v7_fresh_round2/reduction.pkl ['final_expr']
  B = v6   round5  : results/pentagonbox_8_5_v6_round5/reduction.pkl       ['final_expr']

Both start from the SAME integral I[1,1,1,1,1,1,1,1,-5,0,0] and (in paper-masters-only
mode) reduce all corners to the 61 Kira paper masters. If both original reductions
(fresh vs round1-4 gold) were correct, the two final 61-master coefficient vectors must
be IDENTICAL -- proving the earlier 253-vs-262 / 42-paper-coeff differences were purely
symmetry/scaleless (vanishing) relations among corners.

Run:
  python scripts/eval/compare_pmo_final_exprs.py > <log> 2>&1
"""
import os
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_prime, set_paper_masters_only,
                            is_master, weight)

PRIME = 1009
A_PATH = os.path.join(BASE, 'results/pentagonbox_8_5_v7_fresh_round2/reduction.pkl')
B_PATH = os.path.join(BASE, 'results/pentagonbox_8_5_v6_round5/reduction.pkl')

init_from_topology(Topology.from_dir(os.path.join(BASE, 'topology_input/pentagonbox')))
set_prime(PRIME)
set_paper_masters_only(True)   # 61 paper masters only
MS = ibp_env.MASTERS_SET


def load_final(path):
    r = pickle.load(open(path, 'rb'))
    return {tuple(k): v % PRIME for k, v in r['final_expr'].items() if v % PRIME}


A = load_final(A_PATH)
B = load_final(B_PATH)

def summarize(name, expr):
    paper = sum(1 for k in expr if k in MS)
    nonm = sum(1 for k in expr if not is_master(k))
    print(f'{name}: {len(expr)} terms = {paper} PAPER masters + {nonm} NON-master(s)')
    if nonm:
        for k in expr:
            if not is_master(k):
                print(f'    NON-master leftover I{list(k)} = {expr[k]}  w={weight(k)[:2]}')

summarize('A  fresh round2 ', A)
summarize('B  v6   round5  ', B)

# Coefficient-by-coefficient diff over the union of master keys.
keys = set(A) | set(B)
only_A = sorted(set(A) - set(B))
only_B = sorted(set(B) - set(A))
diff = sorted([k for k in (set(A) & set(B)) if A[k] != B[k]])

print(f'\nUnion of master keys: {len(keys)}')
print(f'  only in A (fresh): {len(only_A)}')
print(f'  only in B (v6)   : {len(only_B)}')
print(f'  shared, coeff differs: {len(diff)}')
for k in only_A[:10]:
    print(f'    A-only  I{list(k)} = {A[k]}')
for k in only_B[:10]:
    print(f'    B-only  I{list(k)} = {B[k]}')
for k in diff[:10]:
    print(f'    DIFF    I{list(k)} A={A[k]} B={B[k]}')

identical = (not only_A) and (not only_B) and (not diff)
print('\n' + ('=' * 70))
if identical:
    print('IDENTICAL: both reductions give the SAME 61-paper-master answer.')
    print('=> the original fresh vs round1-4 difference was purely vanishing')
    print('   (symmetry/scaleless) relations among corner integrals. Both correct.')
else:
    print('NOT IDENTICAL: the two paper-master answers differ (see above).')
    print('=> at least one reduction disagrees on the true master coefficients.')
print('=' * 70)
sys.exit(0 if identical else 1)
