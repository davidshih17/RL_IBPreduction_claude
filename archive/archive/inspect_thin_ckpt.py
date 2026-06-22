#!/usr/bin/env python
"""Quick inspector: print beam survivor metadata + path stats."""
import argparse
import gzip
import pickle
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--ranks', type=int, default=3)
    args = p.parse_args()

    with gzip.open(args.ckpt, 'rb') as f:
        d = pickle.load(f)

    print(f'step={d["step"]} beam_size={len(d["beam"])} '
          f'best_weight_ever={d["best_weight_ever"]}')
    print(f'initial_weight={d["initial_weight"]}')
    print(f'target_sector={d["target_sector"]}')
    print(f'start_expr (size={len(d["start_expr"])}):')
    for k, v in list(d['start_expr'].items())[:5]:
        print(f'  {k}: {v}')

    beam = sorted(d['beam'], key=lambda s: (tuple(s['max_w']),
                                             int(s['n_non_masters']),
                                             -float(s['score'])))
    for r in range(min(args.ranks, len(beam))):
        s = beam[r]
        print(f'\n--- rank {r} ---')
        print(f'  max_w={s["max_w"]} nm={s["n_non_masters"]} '
              f'score={s["score"]:.4f} path_len={s["path_len"]}')
        path = s['path']
        # Distinct targets in path
        targets = set(tuple(p[0]) for p in path)
        ibp_ops = set(int(p[1]) for p in path)
        print(f'  unique_targets_in_path={len(targets)} '
              f'unique_ibp_ops={len(ibp_ops)}')
        # First and last few actions
        for i in [0, 1, 2, len(path)//2, len(path)-2, len(path)-1]:
            if 0 <= i < len(path):
                print(f'  path[{i}]: target={path[i][0]} op={path[i][1]} '
                      f'shift={path[i][2]}')


if __name__ == '__main__':
    sys.exit(main())
