"""Exact beam-trajectory equality between two result.pkl files.

For a PARTIAL run (stopped at max-steps, not solved) the final-expr compare is
not meaningful, so this checks the stronger thing for an enumerate-parallelism
change: the two runs produced the byte-identical beam — same length, and for each
slot the same expr, path, score, and n_non_masters. If the fork-pool reorders or
drops any task, this diverges immediately.

Usage: python compare_beams_exact.py <a/result.pkl> <b/result.pkl>
"""
import os
import pickle
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts', 'eval'))


def load_beam(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'beam' not in obj:
        sys.exit(f'{path}: no "beam" key (keys={list(obj) if isinstance(obj, dict) else type(obj)})')
    return obj['beam']


def state_key(sd):
    # sd is a State_v5._asdict(). Use the trajectory-defining fields only;
    # aux_flat is a derived cache and is deliberately excluded.
    expr = sd.get('expr')
    expr_k = tuple(sorted(expr.items())) if isinstance(expr, dict) else expr
    path = sd.get('path')
    path_k = tuple(path) if path is not None else None
    score = sd.get('score')
    return (expr_k, path_k, score, sd.get('n_non_masters'))


def main():
    a, b = sys.argv[1], sys.argv[2]
    ba, bb = load_beam(a), load_beam(b)
    print(f'A: {a}  beam={len(ba)}')
    print(f'B: {b}  beam={len(bb)}')
    if len(ba) != len(bb):
        print(f'==> BEAMS DIFFER: length {len(ba)} != {len(bb)}')
        sys.exit(1)
    nmis = 0
    for i, (sa, sb) in enumerate(zip(ba, bb)):
        ka, kb = state_key(sa), state_key(sb)
        if ka != kb:
            nmis += 1
            if nmis <= 3:
                print(f'  slot {i} DIFFERS:\n    A={ka}\n    B={kb}')
    identical = (nmis == 0)
    print(f'==> BEAMS IDENTICAL: {identical}  ({len(ba)} slots, {nmis} mismatches)')
    sys.exit(0 if identical else 1)


if __name__ == '__main__':
    main()
