#!/usr/bin/env python
"""All-in-one probe comparison: timing + memory + bit-identicality.

Given two probe directories (each containing probe.out and probe.log),
extract per-step (max_w, nm, total time, memory) and print:
  - Common step range
  - Bit-identicality count (max_w + nm match)
  - First divergence step (if any)
  - Side-by-side per-stride table (default every 10 steps)
  - Cumulative timing comparison
  - Memory growth rates
  - Final peak memory + total wall time

Memory comes from condor probe.log event 006 (ResidentSetSize in KB)
correlated to step end-times via probe.out 'total=Ts' cumulative sums.
"""
import argparse
import re
import sys
from datetime import datetime


CONDOR_EXEC_RE = re.compile(
    r'^001 \(\d+\.\d+\.\d+\) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) Job executing'
)
CONDOR_RSS_RE = re.compile(
    r'^006 \(\d+\.\d+\.\d+\) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) Image size'
)
CONDOR_RSS_VAL_RE = re.compile(
    r'^\s*(\d+)\s+-\s+ResidentSetSize of job \(KB\)'
)
STEP_RE = re.compile(
    r'^Step (\d+): .*?P4\(survivor_materialize:.*?aux=([\d.]+)=([\d.]+)s\)'
    r'.*?total=([\d.]+)s tasks=(\d+) cands=(\d+) '
    r'best_max_w=\((\d+), (\d+)\) nm=(\d+)'
)
PEAK_RE = re.compile(r'peak memory: ([\d.]+) MB.*?steps=(\d+)')


def parse_condor_log(path):
    """Return (start_time, [(elapsed_s_from_start, mem_MB), ...])"""
    start = None
    samples = []
    pending_t = None
    try:
        with open(path) as f:
            for line in f:
                m = CONDOR_EXEC_RE.search(line)
                if m and start is None:
                    start = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    continue
                m = CONDOR_RSS_RE.search(line)
                if m:
                    pending_t = datetime.strptime(
                        m.group(1), '%Y-%m-%d %H:%M:%S')
                    continue
                if pending_t is not None:
                    m = CONDOR_RSS_VAL_RE.match(line)
                    if m:
                        rss_kb = int(m.group(1))
                        if start:
                            elapsed = int((pending_t - start).total_seconds())
                            samples.append((elapsed, rss_kb / 1024.0))
                        pending_t = None
    except FileNotFoundError:
        return None, []
    return start, samples


def parse_probe_out(path):
    """Return ([(step, aux_s, total_s, tasks, cands, r, s, nm), ...],
              peak_mb, peak_step)"""
    rows = []
    peak_mb = None
    peak_step = None
    try:
        with open(path) as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    rows.append((
                        int(m.group(1)), float(m.group(2)),
                        float(m.group(4)),
                        int(m.group(5)), int(m.group(6)),
                        int(m.group(7)), int(m.group(8)), int(m.group(9))
                    ))
                m = PEAK_RE.search(line)
                if m:
                    peak_mb = float(m.group(1))
                    peak_step = int(m.group(2))
    except FileNotFoundError:
        pass
    return rows, peak_mb, peak_step


def build_step_to_mem(steps, samples):
    """Map step → mem_MB via cumulative-time correlation."""
    if not samples:
        return {}
    out = {}
    elapsed = 0.0
    j = 0
    for step, aux, total, tasks, cands, r, s, nm in steps:
        elapsed += total
        while j + 1 < len(samples) and samples[j + 1][0] <= elapsed:
            j += 1
        out[step] = (elapsed, samples[j][1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('probe_a', help='Probe directory A (must contain probe.out + probe.log)')
    p.add_argument('probe_b', help='Probe directory B (same)')
    p.add_argument('--label-a', default='A')
    p.add_argument('--label-b', default='B')
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--max-divergences', type=int, default=5)
    args = p.parse_args()

    out_a, peak_a, peak_step_a = parse_probe_out(f'{args.probe_a}/probe.out')
    out_b, peak_b, peak_step_b = parse_probe_out(f'{args.probe_b}/probe.out')
    _, mem_a = parse_condor_log(f'{args.probe_a}/probe.log')
    _, mem_b = parse_condor_log(f'{args.probe_b}/probe.log')

    step_mem_a = build_step_to_mem(out_a, mem_a)
    step_mem_b = build_step_to_mem(out_b, mem_b)

    by_a = {row[0]: row for row in out_a}
    by_b = {row[0]: row for row in out_b}
    common = sorted(set(by_a) & set(by_b))
    print(f'=== {args.label_a} vs {args.label_b} ===')
    print(f'{args.label_a}: {len(out_a)} steps (max={max(by_a) if by_a else 0}) '
          f'peak={peak_a}MB@step{peak_step_a}')
    print(f'{args.label_b}: {len(out_b)} steps (max={max(by_b) if by_b else 0}) '
          f'peak={peak_b}MB@step{peak_step_b}')
    print(f'Common: {len(common)} steps\n')

    if not common:
        return 0

    bit_match = 0
    bit_diff = []
    for s in common:
        ra, rb = by_a[s], by_b[s]
        if ra[5] == rb[5] and ra[6] == rb[6] and ra[7] == rb[7]:
            bit_match += 1
        else:
            bit_diff.append((s, ra, rb))
    first_div = bit_diff[0][0] if bit_diff else None
    print(f'Bit-identical (r,s,nm): {bit_match}/{len(common)}  '
          f'first_divergence_step={first_div}')
    if bit_diff:
        print(f'Divergences (first {args.max_divergences}):')
        for s, ra, rb in bit_diff[:args.max_divergences]:
            print(f'  step {s}: A=(r={ra[5]},s={ra[6]},nm={ra[7]}) '
                  f'B=(r={rb[5]},s={rb[6]},nm={rb[7]})')

    # Per-stride side-by-side table
    print(f'\nPer-{args.stride}-step trajectory (A=={args.label_a}, B=={args.label_b}):')
    print(f'{"step":>4}  '
          f'{"A_max_w":>7} {"A_nm":>4} {"A_t":>6} {"A_mem":>7}   '
          f'{"B_max_w":>7} {"B_nm":>4} {"B_t":>6} {"B_mem":>7}')
    for s in common:
        if s % args.stride != 0 and s != 1 and s != common[-1]:
            continue
        ra, rb = by_a[s], by_b[s]
        ea, ma = step_mem_a.get(s, (0.0, 0.0))
        eb, mb = step_mem_b.get(s, (0.0, 0.0))
        print(f'{s:>4}  '
              f'({ra[5]},{ra[6]}) {ra[7]:>4} {ea:>6.0f} {ma:>6.0f}MB   '
              f'({rb[5]},{rb[6]}) {rb[7]:>4} {eb:>6.0f} {mb:>6.0f}MB')

    # Aggregate timing
    ta = sum(by_a[s][2] for s in common)
    tb = sum(by_b[s][2] for s in common)
    print(f'\nCumulative timing (common steps):')
    print(f'  {args.label_a}: {ta:.1f}s  ({ta/len(common):.2f}s/step)')
    print(f'  {args.label_b}: {tb:.1f}s  ({tb/len(common):.2f}s/step)')
    print(f'  A/B ratio: {ta/max(tb,1e-9):.2f}x')

    # Aux timing
    ax = sum(by_a[s][1] for s in common)
    bx = sum(by_b[s][1] for s in common)
    print(f'  A aux: {ax:.1f}s  ({ax/len(common):.2f}s/step)')
    print(f'  B aux: {bx:.1f}s  ({bx/len(common):.2f}s/step)')
    print(f'  A/B aux ratio: {ax/max(bx,1e-9):.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
