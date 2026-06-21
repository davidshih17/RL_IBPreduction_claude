#!/usr/bin/env python3
"""Compare two HTCondor userlogs by MemoryUsage at equal ELAPSED time since each
job's '001 executing' event. Uses the periodic '006 Image size of job updated'
events (MemoryUsage of job (MB) = cgroup-based). Lets us compare fork-ON vs
fork-OFF on the same integral without waiting for both to finish.

Usage: compare_condor_logs.py <logA> <labelA> <logB> <labelB>
"""
import datetime
import re
import sys


def parse(path):
    """-> (start_dt, [(elapsed_sec, mem_MB, rss_KB), ...])."""
    start = None
    samples = []
    pend_t = None
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        m = re.match(r'^00\d \(.*?\) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', ln)
        if m and ' Job executing on host' in ln:
            start = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        if '006 (' in ln and 'Image size of job updated' in ln:
            t = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', ln)
            pend_t = datetime.datetime.strptime(t.group(1), '%Y-%m-%d %H:%M:%S') if t else None
        mm = re.match(r'\s*(\d+)\s+-\s+MemoryUsage of job \(MB\)', ln)
        if mm and pend_t and start:
            mem = int(mm.group(1))
            rss = None
            if i + 1 < len(lines):
                rm = re.match(r'\s*(\d+)\s+-\s+ResidentSetSize of job \(KB\)', lines[i + 1])
                if rm:
                    rss = int(rm.group(1))
            samples.append(((pend_t - start).total_seconds(), mem, rss))
    return start, samples


def at_elapsed(samples, sec):
    """MemoryUsage (MB) at-or-before `sec` elapsed (step-hold), else None."""
    best = None
    for e, mem, _ in samples:
        if e <= sec + 1:
            best = mem
    return best


logA, labA, logB, labB = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sA, datA = parse(logA)
sB, datB = parse(logB)
print(f'{labA}: start {sA}, {len(datA)} samples, last elapsed '
      f'{datA[-1][0]/60:.0f} min, peak {max(d[1] for d in datA)} MB')
print(f'{labB}: start {sB}, {len(datB)} samples, last elapsed '
      f'{datB[-1][0]/60:.0f} min, peak {max(d[1] for d in datB)} MB')
# Compare over the elapsed range both cover.
maxmin = int(min(datA[-1][0], datB[-1][0]) / 60)
print(f'\n{"elapsed_min":>11} | {labA+" MB":>14} | {labB+" MB":>14} | ratio A/B')
for mn in range(0, maxmin + 1, 2):
    a = at_elapsed(datA, mn * 60)
    b = at_elapsed(datB, mn * 60)
    r = f'{a/b:.1f}x' if (a and b) else '-'
    print(f'{mn:11d} | {str(a):>14} | {str(b):>14} | {r:>8}')
print(f'\npeak over shared window ({maxmin} min): '
      f'{labA}={max(d[1] for d in datA if d[0] <= maxmin*60)} MB  '
      f'{labB}={max(d[1] for d in datB if d[0] <= maxmin*60)} MB')
