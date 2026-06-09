"""Inspect beam diversity in a v5 ckpt.
For each state, compute:
  - expr_fp = frozenset(expr.items())
  - rs_fp   = frozenset((k, frozenset(v.items())) for k, v in resolved_subs.items())
  - path_fp = tuple(path)
Count uniques and report the duplicate-cluster distribution.
"""
import pickle, sys
from collections import Counter

ckpts = sys.argv[1:]
for c in ckpts:
    with open(c, 'rb') as f:
        d = pickle.load(f)
    beam = d['beam']
    n = len(beam)

    expr_fps = []
    expr_rs_fps = []
    path_fps = []
    for s in beam:
        e_fp = frozenset(s['expr'].items())
        rs_fp = frozenset(
            (k, frozenset(v.items())) for k, v in s['resolved_subs'].items()
        )
        expr_fps.append(e_fp)
        expr_rs_fps.append((e_fp, rs_fp))
        path_fps.append(tuple(s.get('path') or ()))

    print(f'\n=== {c}  step={d["step"]}  n_beam={n} ===')
    print(f'  unique expr            : {len(set(expr_fps))} / {n}')
    print(f'  unique (expr, RS)      : {len(set(expr_rs_fps))} / {n}')
    print(f'  unique path            : {len(set(path_fps))} / {n}')

    e_dist = Counter(expr_fps)
    er_dist = Counter(expr_rs_fps)
    print(f'  expr cluster sizes (top): '
          f'{sorted(e_dist.values(), reverse=True)[:10]}')
    print(f'  (expr,RS) cluster sizes : '
          f'{sorted(er_dist.values(), reverse=True)[:10]}')
