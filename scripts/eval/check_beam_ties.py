#!/usr/bin/env python
"""For step 166 and step 167, group survivors by:
  (a) (max_w, n_non_masters) — first two sort keys
  (b) (expr_key, subs_key) — actual state
  (c) (expr_key, subs_key, resolved_subs_key) — fully-specified state

Show how many distinct ties exist, and whether tied children share
(expr, subs) but differ only in resolved_subs / path (= different histories
arriving at the same state).
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')


def expr_key(e):
    return tuple(sorted(e.items()))


def subs_key(s):
    return tuple((k, tuple(sorted(v.items()))) for k, v in sorted(s.items()))


def rs_key(rs):
    return tuple((k, tuple(sorted(v.items()))) for k, v in sorted(rs.items()))


def summarize(beam, label):
    n = len(beam)
    by_mw = Counter()
    by_mw_nm = Counter()
    by_state = defaultdict(list)  # (expr_key, subs_key) -> list of survivor idx
    by_state_rs = defaultdict(list)  # (expr_key, subs_key, rs_key)
    for i, s in enumerate(beam):
        mw = tuple(s['max_w'])
        nm = int(s['n_non_masters'])
        by_mw[mw] += 1
        by_mw_nm[(mw, nm)] += 1
        ek = expr_key(s['expr'])
        sk = subs_key(s['subs'])
        rk = rs_key(s['resolved_subs'])
        by_state[(ek, sk)].append(i)
        by_state_rs[(ek, sk, rk)].append(i)
    print(f'\n=== {label}: {n} survivors ===')
    print(f'  Distinct max_w values: {len(by_mw)}')
    for mw, c in sorted(by_mw.items()):
        print(f'    max_w={mw}: {c}')
    print(f'  Distinct (max_w, n_non_masters) groups (forces score tiebreaker if >1): {len(by_mw_nm)}')
    for k, c in sorted(by_mw_nm.items()):
        if c > 1:
            print(f'    (mw={k[0]}, nm={k[1]}): {c} tied -> score breaks tie')
    # Also check expr alone
    by_expr_only = defaultdict(list)
    by_subs_only = defaultdict(list)
    by_rs_only = defaultdict(list)
    for i, s in enumerate(beam):
        by_expr_only[expr_key(s['expr'])].append(i)
        by_subs_only[subs_key(s['subs'])].append(i)
        by_rs_only[rs_key(s['resolved_subs'])].append(i)
    print(f'  Distinct expr alone:     {len(by_expr_only)}')
    print(f'  Distinct subs alone:     {len(by_subs_only)}')
    print(f'  Distinct RS alone:       {len(by_rs_only)}')
    print(f'  Distinct (expr, subs) states: {len(by_state)}')
    print(f'  Distinct (expr, subs, RS) states: {len(by_state_rs)}')
    # For the largest expr group, show how they differ
    if by_expr_only:
        ek_top, idxs = max(by_expr_only.items(), key=lambda kv: len(kv[1]))
        if len(idxs) > 1:
            print(f'  Largest expr group has {len(idxs)} survivors:')
            distinct_subs = len(set(subs_key(beam[i]['subs']) for i in idxs))
            distinct_rs = len(set(rs_key(beam[i]['resolved_subs']) for i in idxs))
            print(f'    distinct subs within: {distinct_subs}')
            print(f'    distinct RS within:   {distinct_rs}')
            scores = sorted(beam[i]['score'] for i in idxs)
            print(f'    score range: {scores[0]:.6f} .. {scores[-1]:.6f} '
                  f'(spread={scores[-1]-scores[0]:.6f})')
    # How many states are reached by multiple survivors?
    multi_state = [(k, v) for k, v in by_state.items() if len(v) > 1]
    multi_state_rs = [(k, v) for k, v in by_state_rs.items() if len(v) > 1]
    print(f'  (expr, subs) reached by >1 survivor: {len(multi_state)}')
    print(f'  (expr, subs, RS) reached by >1 survivor: {len(multi_state_rs)}')
    if multi_state:
        for (ek, sk), idxs in multi_state[:5]:
            scores = [beam[i]['score'] for i in idxs]
            rs_distinct = len(set(rs_key(beam[i]['resolved_subs']) for i in idxs))
            print(f'    idxs={idxs}  scores={[f"{x:.4f}" for x in scores]}  '
                  f'#distinct_RS={rs_distinct}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_dir')
    args = p.parse_args()

    for step in (166, 167):
        with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step{step:04d}', 'rb') as f:
            c = pickle.load(f)
        summarize(c['beam'], f'step {step}')


if __name__ == '__main__':
    sys.exit(main())
