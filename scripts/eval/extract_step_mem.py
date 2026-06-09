#!/usr/bin/env python
"""Correlate condor .log memory events with probe.out step lines.

Condor event 006 has wall-clock timestamps + ResidentSetSize_KB updates
every minute. probe.out has 'Step N: ... total=Ts' lines giving per-step
duration. We reconstruct step end-time = job_start_time + cumsum(total)
and look up the most recent ResidentSetSize sample at that time.

Output: tab-separated 'step  total_elapsed_s  mem_MB'
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
    r'^Step (\d+): .*?total=([\d.]+)s.*?best_max_w=\((\d+), (\d+)\) nm=(\d+)'
)


def parse_condor_log(path):
    """Return (start_time, [(time, rss_mb), ...])"""
    start = None
    samples = []
    pending_t = None
    with open(path) as f:
        for line in f:
            m = CONDOR_EXEC_RE.search(line)
            if m and start is None:
                start = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                continue
            m = CONDOR_RSS_RE.search(line)
            if m:
                pending_t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                continue
            if pending_t is not None:
                m = CONDOR_RSS_VAL_RE.match(line)
                if m:
                    rss_kb = int(m.group(1))
                    samples.append((pending_t, rss_kb / 1024.0))
                    pending_t = None
    return start, samples


def parse_probe_out(path):
    """Return [(step, total_s, max_w_r, max_w_s, nm), ...]"""
    out = []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)),
                            int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('condor_log')
    p.add_argument('probe_out')
    p.add_argument('--stride', type=int, default=10,
                   help='Print every Nth step (default 10)')
    args = p.parse_args()

    start, samples = parse_condor_log(args.condor_log)
    steps = parse_probe_out(args.probe_out)

    if start is None or not samples or not steps:
        print(f'Parsing failed: start={start} '
              f'#samples={len(samples)} #steps={len(steps)}',
              file=sys.stderr)
        return 1

    # Build sorted (epoch_s_from_start, mem_MB) array for binary lookup.
    sample_es = [(int((t - start).total_seconds()), m) for t, m in samples]

    elapsed = 0.0
    print('step\telapsed_s\tmem_MB\tmax_w\tnm')
    last_mem = sample_es[0][1] if sample_es else 0.0
    j = 0  # cursor into sample_es
    for step, total, r, s, nm in steps:
        elapsed += total
        # advance cursor to latest sample <= elapsed
        while j + 1 < len(sample_es) and sample_es[j + 1][0] <= elapsed:
            j += 1
        last_mem = sample_es[j][1]
        if step % args.stride == 0 or step == 1 or step == steps[-1][0]:
            print(f'{step}\t{elapsed:.1f}\t{last_mem:.1f}\t({r},{s})\t{nm}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
