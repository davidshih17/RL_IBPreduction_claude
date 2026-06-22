#!/usr/bin/env python
"""Diff two probe.out logs step-by-step on best_max_w/nm/total/tasks.

Inputs: two probe.out files (typically v4-rescue vs baseline-1cpu).
Output: per-step deltas + summary (matched range, first divergence, etc.).
"""
import argparse
import re
import sys


STEP_RE = re.compile(
    r'^Step (\d+): .*?P4\(survivor_materialize:.*?aux=([\d.]+)=([\d.]+)s\).*?'
    r'total=([\d.]+)s tasks=(\d+) cands=(\d+) best_max_w=\((\d+), (\d+)\) nm=(\d+)'
)


def parse(path):
    rows = {}
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                rows[step] = {
                    'aux': float(m.group(2)),
                    'p4_total': float(m.group(3)),
                    'total': float(m.group(4)),
                    'tasks': int(m.group(5)),
                    'cands': int(m.group(6)),
                    'r': int(m.group(7)),
                    's': int(m.group(8)),
                    'nm': int(m.group(9)),
                }
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_a', help='v4-rescue probe.out')
    p.add_argument('log_b', help='baseline-1cpu probe.out')
    p.add_argument('--max-print', type=int, default=10)
    args = p.parse_args()

    a = parse(args.log_a)
    b = parse(args.log_b)
    common = sorted(set(a) & set(b))
    a_only = sorted(set(a) - set(b))
    b_only = sorted(set(b) - set(a))
    print(f'A steps: {len(a)} (max={max(a) if a else 0})')
    print(f'B steps: {len(b)} (max={max(b) if b else 0})')
    print(f'Common steps: {len(common)}')

    if not common:
        print('No common steps')
        return 0

    bit_match = 0
    bit_diff = []
    timing = {'a_sum_total': 0.0, 'b_sum_total': 0.0,
              'a_sum_aux': 0.0, 'b_sum_aux': 0.0}
    for s in common:
        ra, rb = a[s], b[s]
        if ra['r'] == rb['r'] and ra['s'] == rb['s'] and ra['nm'] == rb['nm']:
            bit_match += 1
        else:
            bit_diff.append((s, ra, rb))
        timing['a_sum_total'] += ra['total']
        timing['b_sum_total'] += rb['total']
        timing['a_sum_aux'] += ra['aux']
        timing['b_sum_aux'] += rb['aux']

    first_diverge = bit_diff[0][0] if bit_diff else None
    print(f'\nBit-identical (max_w + nm): {bit_match}/{len(common)} '
          f'(first divergence at step {first_diverge})')
    if bit_diff:
        print(f'Divergences (first {args.max_print}):')
        for s, ra, rb in bit_diff[:args.max_print]:
            print(f'  step {s:4d}: A=(r={ra["r"]},s={ra["s"]},nm={ra["nm"]}) '
                  f'B=(r={rb["r"]},s={rb["s"]},nm={rb["nm"]})')

    print(f'\nTiming (common steps):')
    print(f'  A total: {timing["a_sum_total"]:.1f}s  '
          f'(avg {timing["a_sum_total"]/len(common):.2f}s/step)')
    print(f'  B total: {timing["b_sum_total"]:.1f}s  '
          f'(avg {timing["b_sum_total"]/len(common):.2f}s/step)')
    ratio = timing['a_sum_total'] / max(timing['b_sum_total'], 1e-9)
    print(f'  A/B total ratio: {ratio:.2f}x')
    print(f'  A aux: {timing["a_sum_aux"]:.1f}s  '
          f'(avg {timing["a_sum_aux"]/len(common):.2f}s/step)')
    print(f'  B aux: {timing["b_sum_aux"]:.1f}s  '
          f'(avg {timing["b_sum_aux"]/len(common):.2f}s/step)')
    aux_ratio = timing['a_sum_aux'] / max(timing['b_sum_aux'], 1e-9)
    print(f'  A/B aux ratio: {aux_ratio:.2f}x')

    return 0


if __name__ == '__main__':
    sys.exit(main())
