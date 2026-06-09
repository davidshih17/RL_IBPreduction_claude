"""Load saved replay state, classify active_expr terms using the REAL
is_master() (with --no-paper-masters-only semantics = supermaster set =
MASTERS_SET ∪ corners-of-non-master-sectors), and tabulate the true
non-supermaster integrals by (L, r, s) and failure category.
"""
import argparse, os, pickle, sys
from collections import defaultdict, Counter

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')

from sailir import ibp_env
from sailir.ibp_env import (
    init_from_topology, set_prime, set_paper_masters_only, is_master,
)
from sailir.topology import Topology


TOPO_DIR = ('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/'
            'topology_input/pentagonbox')


def Lrs(ig):
    L = sum(1 for x in ig[:8] if x > 0)
    r = sum(x for x in ig if x > 0)
    s = -sum(x for x in ig if x < 0)
    return L, r, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('state_pkl')
    args = p.parse_args()

    # Load topology under --no-paper-masters-only semantics (matches the sweep)
    topo = Topology.from_dir(TOPO_DIR)
    set_prime(1009)
    set_paper_masters_only(False)
    init_from_topology(topo)
    print(f'topology = {ibp_env.FAMILY_NAME},  MASTERS_SET size = '
          f'{len(ibp_env.MASTERS_SET)}')

    state = pickle.load(open(args.state_pkl, 'rb'))
    cache = state['cache']
    expr = state['active_expr']
    log_set = state['log_integrals']
    print(f'active_expr terms : {len(expr)}')

    # Classify each term
    super_master = []
    true_nm = []
    for ig in expr:
        if is_master(ig):
            super_master.append(ig)
        else:
            true_nm.append(ig)

    print(f'  supermasters    : {len(super_master)}')
    print(f'  true non-masters: {len(true_nm)}')
    print()

    # Bucket true non-masters by category and (L,r,s)
    cat_tot = Counter()
    L_tot = Counter()
    by_L_cat = defaultdict(Counter)
    by_Lrs = defaultdict(Counter)
    for ig in true_nm:
        L, r, s = Lrs(ig)
        if ig in cache:
            cat = 'TABU' if cache[ig] == {ig: 1} else 'CACHED_REAL'
        elif ig in log_set:
            cat = 'DIED'
        else:
            cat = 'NEVER'
        cat_tot[cat] += 1
        L_tot[L] += 1
        by_L_cat[L][cat] += 1
        by_Lrs[(L, r, s)][cat] += 1

    print('=== true non-masters by category ===')
    for k in sorted(cat_tot, key=lambda c: -cat_tot[c]):
        print(f'  {k:<14s} {cat_tot[k]}')
    print()
    print('=== true non-masters by level L ===')
    print(f'  {"L":>2s} {"total":>6s} {"TABU":>6s} {"DIED":>6s} {"NEVER":>6s}')
    for L in sorted(L_tot, key=lambda x: -x):
        c = by_L_cat[L]
        print(f'  {L:>2d} {L_tot[L]:>6d} {c.get("TABU",0):>6d} '
              f'{c.get("DIED",0):>6d} {c.get("NEVER",0):>6d}')
    print()
    print('=== true non-masters by (L, r, s) ===')
    print(f'{"L":>2s} {"r":>3s} {"s":>3s} {"total":>6s} '
          f'{"TABU":>6s} {"DIED":>6s} {"NEVER":>6s}')
    for k in sorted(by_Lrs, key=lambda t: (-t[0], -t[1], -t[2])):
        L, r, s = k
        c = by_Lrs[k]
        tot = sum(c.values())
        print(f'{L:>2d} {r:>3d} {s:>3d} {tot:>6d} '
              f'{c.get("TABU",0):>6d} {c.get("DIED",0):>6d} '
              f'{c.get("NEVER",0):>6d}')


if __name__ == '__main__':
    main()
