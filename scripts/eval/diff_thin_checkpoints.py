#!/usr/bin/env python
"""Diff two thin checkpoints to find first survivor-level divergence.

Each checkpoint stores the FULL beam's surviving paths (gzipped pickle).
For step-by-step divergence:
  - Sort each beam by (max_w, n_non_masters, -score) — the production
    --beam-sort weight order.
  - For each rank r ∈ [0, beam_width), compare path[r] step by step.
  - Print the first (rank, step) where the action tuple differs.

Also reports per-rank score / max_w / nm so we can see whether the entire
beam diverged or just specific ranks.

Usage:
  diff_thin_checkpoints.py <ckpt_a> <ckpt_b> [--top N]
"""
import argparse
import gzip
import pickle
import sys


def load_thin(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def sort_key(s):
    return (tuple(s['max_w']), int(s['n_non_masters']), -float(s['score']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_a')
    p.add_argument('ckpt_b')
    p.add_argument('--top', type=int, default=None,
                   help='Only compare top-N survivors (default: all)')
    args = p.parse_args()

    a = load_thin(args.ckpt_a)
    b = load_thin(args.ckpt_b)
    print(f'A: step={a["step"]} beam_size={len(a["beam"])} '
          f'best_weight={a["best_weight_ever"]}')
    print(f'B: step={b["step"]} beam_size={len(b["beam"])} '
          f'best_weight={b["best_weight_ever"]}')

    beam_a = sorted(a['beam'], key=sort_key)
    beam_b = sorted(b['beam'], key=sort_key)
    n = min(len(beam_a), len(beam_b))
    if args.top is not None:
        n = min(n, args.top)
    print(f'Comparing top {n} survivors')

    diverged = False
    for r in range(n):
        sa = beam_a[r]
        sb = beam_b[r]
        pa = sa['path']
        pb = sb['path']
        # First action mismatch (if any)
        L = min(len(pa), len(pb))
        first_mismatch = None
        for i in range(L):
            if tuple(pa[i]) != tuple(pb[i]):
                first_mismatch = i
                break
        if first_mismatch is None and len(pa) == len(pb):
            continue  # rank r is bit-identical
        diverged = True
        print(f'\n--- rank {r} DIVERGED ---')
        print(f'  A: max_w={sa["max_w"]} nm={sa["n_non_masters"]} '
              f'score={sa["score"]:.4f} path_len={len(pa)}')
        print(f'  B: max_w={sb["max_w"]} nm={sb["n_non_masters"]} '
              f'score={sb["score"]:.4f} path_len={len(pb)}')
        if first_mismatch is not None:
            print(f'  First action mismatch at step {first_mismatch}:')
            print(f'    A[{first_mismatch}] = {pa[first_mismatch]}')
            print(f'    B[{first_mismatch}] = {pb[first_mismatch]}')
            # Show 3 prior shared actions for context
            lo = max(0, first_mismatch - 3)
            print(f'  Common prefix (steps {lo}..{first_mismatch-1}):')
            for i in range(lo, first_mismatch):
                print(f'    [{i}] {pa[i]}')
        else:
            print(f'  Common prefix matches; lengths differ '
                  f'(A={len(pa)} B={len(pb)})')
        if r >= 5:
            break  # report up to 6 diverged ranks then stop

    if not diverged:
        print('\nAll compared ranks bit-identical.')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
