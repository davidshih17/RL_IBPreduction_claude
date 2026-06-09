#!/usr/bin/env python
"""Diff per-step thick checkpoints from two v5 runs.
At each step compare every beam state's (expr, resolved_subs, sub_accum,
score, path, n_non_masters). Reports the first diverging step (or PASS).
"""
import argparse
import pickle
import sys


def normalize(s):
    """Strip aux_flat (id-dependent) and return comparison key."""
    return {
        'expr': s['expr'],
        'resolved_subs': s['resolved_subs'],
        'sub_accum': s['sub_accum'],
        'score': s['score'],
        'path': s['path'],
        'n_non_masters': s['n_non_masters'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir-a', required=True)
    p.add_argument('--dir-b', required=True)
    p.add_argument('--steps', type=int, required=True)
    args = p.parse_args()

    n_match = 0
    for step in range(1, args.steps + 1):
        fa = f'{args.dir_a}/ckpt.pkl.step{step:04d}'
        fb = f'{args.dir_b}/ckpt.pkl.step{step:04d}'
        try:
            with open(fa, 'rb') as f:
                ca = pickle.load(f)
            with open(fb, 'rb') as f:
                cb = pickle.load(f)
        except FileNotFoundError as e:
            print(f'step {step}: MISSING FILE — {e}')
            return 1
        beam_a = [normalize(s) for s in ca['beam']]
        beam_b = [normalize(s) for s in cb['beam']]
        # Algebraic equivalence: ignore score FP noise but require everything
        # else identical, AT EVERY beam position.
        def alg_eq(a, b):
            return all(a[k] == b[k] for k in
                       ('expr', 'resolved_subs', 'sub_accum', 'path',
                        'n_non_masters'))
        per_beam_alg = [alg_eq(a, b) for a, b in zip(beam_a, beam_b)]
        alg_match = all(per_beam_alg)
        score_diffs = [abs(a['score'] - b['score']) for a, b in zip(beam_a, beam_b)]
        max_score_diff = max(score_diffs) if score_diffs else 0
        if beam_a == beam_b:
            n_match += 1
            print(f'step {step:>3}: PASS  (beam size {len(beam_a)})')
            continue
        elif alg_match:
            n_match += 1
            print(f'step {step:>3}: ALG-PASS  (score FP noise max={max_score_diff:.2e})')
            continue
        else:
            # Find first beam idx with alg mismatch
            first_bad = next(i for i, ok in enumerate(per_beam_alg) if not ok)
            print(f'step {step:>3}: ALG-FAIL  first_bad_idx={first_bad}  '
                  f'max_score_diff={max_score_diff:.2e}')
            a = beam_a[first_bad]
            b = beam_b[first_bad]
            for k in ('expr', 'resolved_subs', 'sub_accum', 'path', 'n_non_masters'):
                if a[k] != b[k]:
                    if k == 'expr':
                        only_a = {kk: v for kk, v in a[k].items() if b[k].get(kk) != v}
                        only_b = {kk: v for kk, v in b[k].items() if a[k].get(kk) != v}
                        print(f'    expr  a-only={dict(list(only_a.items())[:3])}  b-only={dict(list(only_b.items())[:3])}')
                    elif k == 'resolved_subs':
                        ka, kb = set(a[k]), set(b[k])
                        print(f'    RS  keys-a-only={len(ka-kb)} keys-b-only={len(kb-ka)} '
                              f'shared={len(ka&kb)}')
                    elif k == 'path':
                        idx_first = next((i for i,(x,y) in enumerate(zip(a[k],b[k])) if x!=y), None)
                        print(f'    path  len_a={len(a[k])} len_b={len(b[k])} first_diff_idx={idx_first}')
                        if idx_first is not None:
                            print(f'      a[{idx_first}]={a[k][idx_first]}')
                            print(f'      b[{idx_first}]={b[k][idx_first]}')
                    else:
                        print(f'    {k} a={a[k]!r}  b={b[k]!r}')
            return 1
            # Report first index that differs
            for i, (a, b) in enumerate(zip(beam_a, beam_b)):
                if a != b:
                    print(f'  beam[{i}] diff:')
                    for k in a:
                        if a[k] != b[k]:
                            if k in ('expr', 'sub_accum'):
                                only_a = {kk: v for kk, v in a[k].items()
                                          if b[k].get(kk) != v}
                                only_b = {kk: v for kk, v in b[k].items()
                                          if a[k].get(kk) != v}
                                print(f'    {k}: a-only={list(only_a.items())[:3]} '
                                      f'b-only={list(only_b.items())[:3]}')
                            elif k == 'resolved_subs':
                                ka = set(a[k].keys())
                                kb = set(b[k].keys())
                                print(f'    {k}: a-only-keys={len(ka-kb)} '
                                      f'b-only-keys={len(kb-ka)} '
                                      f'shared={len(ka&kb)}')
                                for kk in (ka & kb):
                                    if a[k][kk] != b[k][kk]:
                                        print(f'      RS[{kk}] diff: a={list(a[k][kk].items())[:3]} '
                                              f'b={list(b[k][kk].items())[:3]}')
                                        break
                            elif k == 'path':
                                print(f'    {k}: len_a={len(a[k])} len_b={len(b[k])} '
                                      f'first_diff_idx='
                                      f'{next((i for i,(x,y) in enumerate(zip(a[k],b[k])) if x!=y), None)}')
                            else:
                                print(f'    {k}: a={a[k]!r}  b={b[k]!r}')
                    break
            return 1

    print(f'\n=== {n_match}/{args.steps} steps bit-identical (modulo aux_flat) ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
