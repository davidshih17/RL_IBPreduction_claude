"""Tabulate active-expr non-masters by (L, r, s) and failure category.
Loads <sweep_root>/replay_state.pkl produced by save_replay_state.py.

Categories:
  TABU              : in cache as identity
  DISPATCHED_DIED   : not in cache, has .log
  NEVER             : not in cache, no .log
"""
import argparse, os, pickle
from collections import defaultdict, Counter


def Lrs(ig):
    L = sum(1 for x in ig[:8] if x > 0)
    r = sum(x for x in ig if x > 0)
    s = -sum(x for x in ig if x < 0)
    return L, r, s


def is_paper_master(ig):
    return all(0 <= x <= 1 for x in ig[:8]) and all(x >= 0 for x in ig[8:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('state_pkl')
    args = p.parse_args()
    state = pickle.load(open(args.state_pkl, 'rb'))
    cache = state['cache']
    expr = state['active_expr']
    log_set = state['log_integrals']

    by_Lrs = defaultdict(Counter)
    cat_tot = Counter()
    L_tot = Counter()
    by_L_cat = defaultdict(Counter)
    n_master = 0
    for ig in expr:
        if is_paper_master(ig):
            n_master += 1
            continue
        L, r, s = Lrs(ig)
        if ig in cache:
            cat = 'TABU' if cache[ig] == {ig: 1} else 'CACHED_REAL'
        elif ig in log_set:
            cat = 'DIED'
        else:
            cat = 'NEVER'
        by_Lrs[(L, r, s)][cat] += 1
        cat_tot[cat] += 1
        L_tot[L] += 1
        by_L_cat[L][cat] += 1

    print(f'active_expr terms : {len(expr)}')
    print(f'  paper masters   : {n_master}')
    print(f'  non-masters     : {sum(cat_tot.values())}')
    print()
    print('=== non-masters by category ===')
    for k in sorted(cat_tot, key=lambda c: -cat_tot[c]):
        print(f'  {k:<6s} {cat_tot[k]}')
    print()
    print('=== non-masters by level L ===')
    print(f'  {"L":>2s} {"total":>6s} {"TABU":>6s} {"DIED":>6s} {"NEVER":>6s}')
    for L in sorted(L_tot, key=lambda x: -x):
        c = by_L_cat[L]
        print(f'  {L:>2d} {L_tot[L]:>6d} {c.get("TABU",0):>6d} '
              f'{c.get("DIED",0):>6d} {c.get("NEVER",0):>6d}')
    print()
    print('=== non-masters by (L, r, s) ===')
    print(f'{"L":>2s} {"r":>3s} {"s":>3s} {"total":>6s} {"TABU":>6s} '
          f'{"DIED":>6s} {"NEVER":>6s}')
    for k in sorted(by_Lrs, key=lambda t: (-t[0], -t[1], -t[2])):
        L, r, s = k
        c = by_Lrs[k]
        tot = sum(c.values())
        print(f'{L:>2d} {r:>3d} {s:>3d} {tot:>6d} '
              f'{c.get("TABU",0):>6d} {c.get("DIED",0):>6d} '
              f'{c.get("NEVER",0):>6d}')


if __name__ == '__main__':
    main()
