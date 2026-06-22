"""Load a tracemalloc snapshot dumped by beam_search_v6_memprofile.py and
print the top-N allocators grouped by file:line, then by traceback.

Usage:
    python analyze_memprofile.py <peak_step*.snap> [--top 30]
"""
import argparse, os, sys, tracemalloc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('snap_path')
    p.add_argument('--top', type=int, default=30)
    args = p.parse_args()

    snap = tracemalloc.Snapshot.load(args.snap_path)
    total = sum(s.size for s in snap.statistics('filename'))
    print(f'file: {args.snap_path}')
    print(f'total traced memory: {total/1e6:.1f} MB')
    print()
    print(f'=== top {args.top} by line ===')
    stats = snap.statistics('lineno')
    for i, s in enumerate(stats[:args.top]):
        f = s.traceback[0]
        print(f'  #{i+1:>3d} {s.size/1e6:>8.1f} MB  {s.count:>10d} blocks  '
              f'{f.filename}:{f.lineno}')

    print()
    print(f'=== top 10 by traceback (full stack) ===')
    stats_tb = snap.statistics('traceback')
    for i, s in enumerate(stats_tb[:10]):
        print(f'\n  #{i+1} {s.size/1e6:.1f} MB  ({s.count} blocks)')
        for line in s.traceback.format():
            print(f'    {line}')


if __name__ == '__main__':
    main()
