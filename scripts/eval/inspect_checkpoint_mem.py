"""Deep-size every component of a beam-search checkpoint to find what's heavy.

Reports: step count, beam width, per-state expr/resolved_subs/aux sizes, and the
deep byte size of beam vs tabu_dict. Helps locate the memory hog. (Note: the
runtime _bt_tabu and env._raw_eq_cache are NOT in the checkpoint — if the
checkpoint is small but RAM was huge, the hog is one of those.)
"""
import pickle
import sys
from collections import deque

import numpy as np


def deep_size(obj, seen=None):
    """Recursive byte size with id-dedup; handles numpy + containers."""
    if seen is None:
        seen = set()
    total = 0
    stack = deque([obj])
    while stack:
        o = stack.popleft()
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(o, np.ndarray):
            total += o.nbytes
            continue
        total += sys.getsizeof(o, 0)
        if isinstance(o, dict):
            for k, v in o.items():
                stack.append(k)
                stack.append(v)
        elif isinstance(o, (list, tuple, set, frozenset, deque)):
            for x in o:
                stack.append(x)
    return total


def mb(b):
    return b / 1e6


def main():
    path = sys.argv[1]
    print(f'loading {path} ...', flush=True)
    with open(path, 'rb') as f:
        ck = pickle.load(f)

    print(f'\ntop-level keys: {list(ck.keys())}')
    print(f'step          : {ck.get("step")}')

    beam = ck.get('beam', [])
    print(f'beam width    : {len(beam)}')

    # tabu_dict in checkpoint
    td = ck.get('tabu_dict')
    if td is None:
        print('tabu_dict     : None (live tabu is the uncheckpointed _bt_tabu)')
    else:
        n_entries = sum(len(v) for _, v in td) if isinstance(td, list) else 0
        print(f'tabu_dict     : {len(td)} expr keys, {n_entries} total tuples, '
              f'deep={mb(deep_size(td)):.1f} MB')

    # Per-state breakdown
    print('\n=== per-state sizes (first 5 of beam) ===')
    expr_terms = []
    rs_lens = []
    aux_sizes = []
    for i, sd in enumerate(beam):
        expr = sd.get('expr', {})
        rs = sd.get('resolved_subs', {})
        aux = sd.get('aux_flat')
        expr_terms.append(len(expr))
        rs_lens.append(len(rs) if rs else 0)
        a_sz = deep_size(aux) if aux is not None else 0
        aux_sizes.append(a_sz)
        if i < 5:
            print(f'  state {i}: expr_terms={len(expr)}  '
                  f'resolved_subs={len(rs) if rs else 0}  '
                  f'expr_deep={mb(deep_size(expr)):.1f}MB  '
                  f'aux_deep={mb(a_sz):.1f}MB')

    print('\n=== beam aggregate ===')
    print(f'  expr terms: min={min(expr_terms)} max={max(expr_terms)} '
          f'mean={sum(expr_terms)//len(expr_terms)}')
    print(f'  resolved_subs: min={min(rs_lens)} max={max(rs_lens)} '
          f'mean={sum(rs_lens)//len(rs_lens)}')
    print(f'  aux_flat total deep: {mb(sum(aux_sizes)):.1f} MB')
    print(f'  beam total deep:     {mb(deep_size(beam)):.1f} MB')
    print(f'  whole checkpoint deep: {mb(deep_size(ck)):.1f} MB')


if __name__ == '__main__':
    main()
