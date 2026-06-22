#!/usr/bin/env python
"""Compare baseline-tabu vs rank-cycle per-task pick dumps.

For each step, load both picks_step<N>.pkl (list of
(expr_fp, target, n_v, picked_actions, V)), key by (expr_fp, target), and
report the first step where the picked-action sets differ, with detail on the
differing tasks (n_v on each side, V, and the symmetric diff of picks).
"""
import os
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
BL = f'{BASE}/results/dbg_picks/baseline'
RC = f'{BASE}/results/dbg_picks/btabu_C60'


def load_step(d, n):
    p = f'{d}/picks_step{n}.pkl'
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        recs = pickle.load(f)
    # key by (expr_fp, target) -> (n_v, picked, V)
    out = {}
    for expr_fp, target, n_v, picked, V in recs:
        out[(expr_fp, target)] = (n_v, picked, V)
    return out


def main():
    first_div_done = False
    for step in range(0, 9):
        bl = load_step(BL, step)
        rc = load_step(RC, step)
        if bl is None or rc is None:
            continue
        bl_keys = set(bl)
        rc_keys = set(rc)
        only_bl = bl_keys - rc_keys
        only_rc = rc_keys - bl_keys
        common = bl_keys & rc_keys
        n_diff_pick = 0
        diff_detail = []
        for k in common:
            bn, bp, bv = bl[k]
            rn, rp, rv = rc[k]
            if bp != rp:
                n_diff_pick += 1
                diff_detail.append((k, bn, rn, bv, rv, bp, rp))
        # Max blocked-count (V field = _n_blocked) seen this step on rc side.
        max_blocked = max((rc[k][2] for k in common), default=-1)
        over_cap = sum(1 for k in common if rc[k][2] > 60)
        print(f'step {step}: tasks bl={len(bl)} rc={len(rc)} '
              f'common={len(common)} only_bl={len(only_bl)} '
              f'only_rc={len(only_rc)} pick_diffs={n_diff_pick} '
              f'max_blocked={max_blocked} tasks_over_cap={over_cap}')
        if (only_bl or only_rc or n_diff_pick) and not first_div_done:
            first_div_done = True
            print(f'  >>> FIRST DIVERGENCE at step {step}')
            for (k, bn, rn, bv, rv, bp, rp) in diff_detail[:5]:
                _expr, _tgt = k
                only_in_bl = set(bp) - set(rp)
                only_in_rc = set(rp) - set(bp)
                print(f'    task target={_tgt}')
                print(f'      n_v: baseline={bn} btabu={rn}   '
                      f'blocked(rc)={rv}  (cap=60)')
                print(f'      in baseline only ({len(only_in_bl)}): '
                      f'{sorted(only_in_bl)[:4]}')
                print(f'      in btabu only    ({len(only_in_rc)}): '
                      f'{sorted(only_in_rc)[:4]}')


if __name__ == '__main__':
    main()
