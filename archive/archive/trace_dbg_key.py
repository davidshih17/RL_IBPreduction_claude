#!/usr/bin/env python
"""Trace the first-divergence (expr_fp, target) key across all dumped steps in
both baseline and rank-cycle, to show how the valid list and picks evolved.
"""
import os
import pickle

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
BL = f'{BASE}/results/dbg_picks/baseline'
RC = f'{BASE}/results/dbg_picks/rankcyc_C60'

# The diverging target from compare_dbg_picks.py
TARGET = (-1, 1, 1, 1, 1, 1, 1, 2, -4, 0, 0)


def load(d, n):
    p = f'{d}/picks_step{n}.pkl'
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def trace(label, d):
    print(f'=== {label} ===')
    for step in range(0, 6):
        recs = load(d, step)
        if recs is None:
            continue
        for expr_fp, target, n_v, picked, V in recs:
            if tuple(target) == TARGET:
                print(f'  step{step}: n_v={n_v} V={V} npick={len(picked)}')
                # show the picked action set as a compact sorted list
                ps = sorted(picked)
                print(f'    picks={ps}')


def main():
    trace('BASELINE', BL)
    trace('RANKCYC', RC)
    # Also: at the divergence step (2), show baseline's removed set vs the
    # full list by comparing baseline n_v to rankcyc n_v.
    print('\n=== cross-check at step 2 ===')
    for label, d in (('baseline', BL), ('rankcyc', RC)):
        recs = load(d, 2)
        for expr_fp, target, n_v, picked, V in recs:
            if tuple(target) == TARGET:
                print(f'  {label}: n_v={n_v} V={V}')


if __name__ == '__main__':
    main()
