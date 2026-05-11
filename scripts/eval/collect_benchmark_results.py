#!/usr/bin/env python3
"""Collect benchmark timing and memory results from async mixed benchmark runs.

Usage:
  python collect_benchmark_results.py           # v13 (default)
  python collect_benchmark_results.py --old     # old (pre-v13) runs
"""

import pickle
import csv
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent.resolve()
# Overridable via CLI; set in main() after argparse.
LOGDIR    = REPO_DIR / 'logs'
RESULTDIR = REPO_DIR / 'results'
SCRATCHDIR = Path.home() / 'scratch'
STRAGGLER_CPUS = 8

# Benchmark integrals in order from the CSV
INTEGRALS = [
    (7, 4, 1, "1,1,2,1,1,1,-4"),
    (8, 4, 2, "2,1,1,1,1,2,-4"),
    (9, 4, 3, "1,1,2,2,1,2,-4"),
    (10, 4, 4, "2,1,2,1,2,2,-4"),
    (10, 5, 4, "1,1,2,2,1,3,-5"),
    (10, 6, 4, "1,1,3,2,2,1,-6"),
    (11, 4, 5, "2,2,2,1,1,3,-4"),
    (11, 5, 5, "1,1,2,3,2,2,-5"),
    (11, 6, 5, "1,4,2,1,2,1,-6"),
    (12, 4, 6, "2,3,1,3,1,2,-4"),
    (12, 5, 6, "1,2,2,2,1,4,-5"),
    (12, 6, 6, "3,2,3,2,1,1,-6"),
    (13, 4, 7, "2,3,3,3,1,1,-4"),
    (13, 5, 7, "2,2,3,3,2,1,-5"),
    (13, 6, 7, "3,2,1,3,2,2,-6"),
    (10, 7, 4, "2,3,1,1,2,1,-7"),
    (11, 7, 5, "2,1,1,2,3,2,-7"),
    (12, 7, 6, "3,1,1,1,1,5,-7"),
    (13, 7, 7, "2,2,3,3,1,2,-7"),
]

def integral_label(integral_str):
    return integral_str.replace(',', '_').replace('-', 'm')

def scan_worker_memory(work_dir):
    """Scan worker result pickles and compute max memory per CPU."""
    results_dir = work_dir / 'results'
    if not results_dir.exists():
        return 0, 0

    max_mem_per_cpu = 0
    max_mem_raw = 0

    for pkl_file in results_dir.glob('*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                result = pickle.load(f)
            mem_kb = result.get('peak_memory_kb', 0)
            if mem_kb == 0:
                continue

            # Determine CPUs from filename: straggler_ prefix means STRAGGLER_CPUS
            fname = pkl_file.stem
            if fname.startswith('straggler_'):
                cpus = STRAGGLER_CPUS
            else:
                cpus = 1

            mem_per_cpu = mem_kb / cpus
            if mem_per_cpu > max_mem_per_cpu:
                max_mem_per_cpu = mem_per_cpu
            if mem_kb > max_mem_raw:
                max_mem_raw = mem_kb
        except Exception:
            continue

    return max_mem_per_cpu, max_mem_raw

def parse_log(log_file):
    """Parse orchestrator log for timing and memory."""
    info = {}
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Time:'):
                # Time: 123.4s (2.1 min)
                info['time_s'] = float(line.split('s')[0].split(':')[1].strip())
            elif line.startswith('Wall clock time:'):
                # Wall clock time: 754.5s (12.6 min)
                info['time_s'] = float(line.split('s')[0].split(':')[1].strip())
            elif line.startswith('Ideal parallel time:'):
                # Ideal parallel time: 625.1s (10.4 min) [critical path...]
                info['ideal_parallel_time_s'] = float(line.split('s')[0].split(':')[1].strip())
            elif line.startswith('Total jobs submitted:'):
                info['total_jobs'] = int(line.split(':')[1].strip())
            elif line.startswith('Stragglers resubmitted:'):
                info['stragglers'] = int(line.split(':')[1].strip())
            elif line.startswith('Total steps:'):
                info['total_steps'] = int(line.split(':')[1].strip())
            elif line.startswith('Cache size:'):
                info['cache_size'] = int(line.split(':')[1].strip())
            elif line.startswith('Cache hits:'):
                info['cache_hits'] = int(line.split(':')[1].strip())
            elif line.startswith('Peak memory (orchestrator):'):
                # Peak memory (orchestrator): 526.3 MB (538924 KB)
                parts = line.split('(')
                info['orchestrator_memory_kb'] = int(parts[-1].replace(' KB)', '').strip())
            elif line.startswith('Peak memory (max worker'):
                # Handles both old "Peak memory (max worker):" and
                # v13 "Peak memory (max worker raw):" / "Peak memory (max worker per-CPU):"
                parts = line.split('(')
                mem_kb = int(parts[-1].replace(' KB)', '').strip())
                if 'per-CPU' in line:
                    info['max_worker_memory_percpu_kb'] = mem_kb
                else:
                    info['max_worker_memory_raw_kb'] = mem_kb
            elif line.startswith('SUCCESS'):
                info['success'] = True
            elif line.startswith('WARNING:') and 'non-masters remaining' in line:
                info['success'] = False
    return info

def main():
    import argparse
    global LOGDIR, RESULTDIR, SCRATCHDIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',     default=str(LOGDIR),
                    help='Directory containing async_mixed_bench_*.log files')
    ap.add_argument('--resultdir',  default=str(RESULTDIR),
                    help='Where to write the summary CSV')
    ap.add_argument('--scratchdir', default=str(SCRATCHDIR),
                    help='Directory containing per-run ibp_async_mixed_bench_* subdirs with worker pickles')
    ap.add_argument('--old', action='store_true', help='Process pre-v13 runs (different filename prefix)')
    args = ap.parse_args()
    LOGDIR     = Path(args.logdir)
    RESULTDIR  = Path(args.resultdir)
    SCRATCHDIR = Path(args.scratchdir)
    use_v13 = not args.old
    prefix = 'v13_' if use_v13 else ''
    version_label = 'v13' if use_v13 else 'old'

    rows = []
    for r, s, d, integral_str in INTEGRALS:
        label = integral_label(integral_str)
        log_file = LOGDIR / f'async_mixed_bench_{prefix}{label}.log'
        work_dir = SCRATCHDIR / f'ibp_async_mixed_bench_{prefix}{label}'

        if not log_file.exists():
            print(f"WARNING: missing log for I[{integral_str}]")
            continue

        info = parse_log(log_file)

        # Scan worker pickles for per-CPU memory
        max_mem_per_cpu_kb, max_mem_raw_kb = scan_worker_memory(work_dir)

        # Use log-reported per-CPU memory if available, else fall back to pickle scan
        worker_percpu_kb = info.get('max_worker_memory_percpu_kb', 0)
        if worker_percpu_kb == 0:
            worker_percpu_kb = max_mem_per_cpu_kb
        worker_raw_kb = info.get('max_worker_memory_raw_kb', 0)
        if worker_raw_kb == 0:
            worker_raw_kb = max_mem_raw_kb

        row = {
            'r': r,
            's': s,
            'd': d,
            'integral': f'I[{integral_str.replace(",", ", ")}]',
            'success': info.get('success', False),
            'time_s': info.get('time_s', 0),
            'ideal_parallel_time_s': info.get('ideal_parallel_time_s', 0),
            'total_jobs': info.get('total_jobs', 0),
            'total_steps': info.get('total_steps', 0),
            'stragglers': info.get('stragglers', 0),
            'cache_size': info.get('cache_size', 0),
            'cache_hits': info.get('cache_hits', 0),
            'orchestrator_memory_mb': info.get('orchestrator_memory_kb', 0) / 1024,
            'max_worker_memory_raw_mb': worker_raw_kb / 1024,
            'max_worker_memory_per_cpu_mb': worker_percpu_kb / 1024,
        }
        rows.append(row)

        status = 'OK' if row['success'] else 'FAIL'
        print(f"  [{status}] {row['integral']:40s}  {row['time_s']:8.1f}s  "
              f"jobs={row['total_jobs']:6d}  steps={row['total_steps']:7d}  "
              f"orch={row['orchestrator_memory_mb']:.0f}MB  "
              f"worker={row['max_worker_memory_per_cpu_mb']:.0f}MB/cpu  "
              f"worker_raw={row['max_worker_memory_raw_mb']:.0f}MB  "
              f"stragglers={row['stragglers']}")

    # Write CSV
    output_csv = RESULTDIR / f'benchmark_mixed_summary_{version_label}.csv'
    fieldnames = list(rows[0].keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to {output_csv}")

if __name__ == '__main__':
    main()
