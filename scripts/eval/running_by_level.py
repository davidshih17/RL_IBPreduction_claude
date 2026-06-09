"""Bucket currently-running condor workers by level L (and optionally by
(L,r,s)). Reads condor_q -af ClusterId Args once.

Usage:
    python running_by_level.py [--lrs]   # also break down by (L,r,s)
"""
import argparse, re, subprocess
from collections import Counter


def parse_integral_from_args(args_str):
    m = re.search(r"--integral=['\"]?([\d,\-]+)", args_str)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split(','))


def level_r_s(integ):
    L = sum(1 for x in integ[:8] if x > 0)
    r = sum(x for x in integ if x > 0)
    s = -sum(x for x in integ if x < 0)
    return L, r, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lrs', action='store_true')
    args = p.parse_args()

    cp = subprocess.run(
        ['condor_q', '-constraint', 'JobStatus==2',
         '-af', 'ClusterId', 'Args'],
        capture_output=True, text=True, timeout=120,
    )
    by_L = Counter()
    by_Lrs = Counter()
    total = 0
    unparsed = 0
    for line in cp.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        integ = parse_integral_from_args(parts[1])
        if integ is None:
            unparsed += 1
            continue
        L, r, s = level_r_s(integ)
        by_L[L] += 1
        by_Lrs[(L, r, s)] += 1
        total += 1

    print(f'Total currently-running workers: {total} (unparsed: {unparsed})')
    print()
    print(f'{"level":>5s} {"count":>8s}')
    for L in sorted(by_L, key=lambda x: -x):
        print(f'  L={L:<2d} {by_L[L]:>7d}')

    if args.lrs:
        print()
        print(f'By (L, r, s):')
        for k in sorted(by_Lrs, key=lambda t: (-t[0], -t[1], -t[2])):
            L, r, s = k
            print(f'  L={L:<2d} r={r:<2d} s={s:<2d}  {by_Lrs[k]}')


if __name__ == '__main__':
    main()
