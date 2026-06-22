"""Explicit per-master coefficient check of the two clean reductions, in the
62-master no-symmetry basis (topology_input/pentagonbox_nosym).

  A = fresh round2 : results/pentagonbox_8_5_v7_fresh_round2/reduction.pkl ['final_expr']
  B = v6   round5  : results/pentagonbox_8_5_v6_round5/reduction.pkl       ['final_expr']

For EVERY one of the 62 masters, prints coeff_A vs coeff_B (0 if absent) and flags
any mismatch. Also asserts neither final_expr contains a non-master (i.e. both are
fully reduced to the 62-master basis).
"""
import os
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, weight

PRIME = 1009
# Use the 62-master no-symmetry basis.
init_from_topology(Topology.from_dir(os.path.join(BASE, 'topology_input/pentagonbox_nosym')))
set_prime(PRIME)
set_paper_masters_only(True)
M62 = ibp_env.MASTERS_SET
print(f'62-master basis loaded: {len(M62)} masters')


def load_final(path):
    r = pickle.load(open(path, 'rb'))
    return {tuple(k): v % PRIME for k, v in r['final_expr'].items() if v % PRIME}


A = load_final(os.path.join(BASE, 'results/pentagonbox_8_5_v7_fresh_round2/reduction.pkl'))
B = load_final(os.path.join(BASE, 'results/pentagonbox_8_5_v6_round5/reduction.pkl'))

# Sanity: neither contains a non-master.
a_nonmaster = [k for k in A if k not in M62]
b_nonmaster = [k for k in B if k not in M62]
print(f'non-master terms in A: {len(a_nonmaster)}   in B: {len(b_nonmaster)}  (must be 0)')

# Per-master table over ALL 62 masters, heaviest first.
masters_sorted = sorted(M62, key=lambda i: (-weight(i)[0], -weight(i)[1], i))
print(f'\n{"#":>3} {"cA":>5} {"cB":>5}  {"=?":>2}  master')
print('-' * 60)
n_nonzero = n_zero = n_mismatch = 0
for n, m in enumerate(masters_sorted, 1):
    ca, cb = A.get(m, 0), B.get(m, 0)
    eq = (ca == cb)
    if not eq:
        n_mismatch += 1
    if ca or cb:
        n_nonzero += 1
        flag = '' if eq else '  <<< MISMATCH'
        print(f'{n:>3} {ca:>5} {cb:>5}  {"Y" if eq else "N":>2}  I{list(m)}{flag}')
    else:
        n_zero += 1
print('-' * 60)
print(f'62 masters: {n_nonzero} nonzero (shown) + {n_zero} zero-in-both')
print(f'coefficient mismatches across all 62 masters: {n_mismatch}')
ok = (n_mismatch == 0 and not a_nonmaster and not b_nonmaster)
print('\n' + ('VERIFIED: all 62 master coefficients are identical in A and B.'
              if ok else 'FAIL: see mismatches above.'))
sys.exit(0 if ok else 1)
