#!/usr/bin/env python3
"""Plot benchmark comparison: Time and Memory vs s, grouped by r.

Dashed lines: Kira. Solid lines: SAILIR. r in {10,11,12,13}.
Reproduces Fig. 3 of the paper.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).parent.parent.parent.resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kira-csv',   default=str(REPO_DIR / 'results/kira_benchmark.csv'))
    ap.add_argument('--sailir-csv', default=str(REPO_DIR / 'results/sailir_benchmark.csv'))
    ap.add_argument('--out',        default=str(REPO_DIR / 'results/benchmark_comparison.pdf'))
    args = ap.parse_args()

    with open(args.kira_csv) as f:
        kira_data = [{
            'r': int(row['r']), 's': int(row['s']),
            'time': float(row['total_time_s']),
            'mem': int(row['peak_memory_kb']) / 1024,
        } for row in csv.DictReader(f)]

    with open(args.sailir_csv) as f:
        ml_data = [{
            'r': int(row['r']), 's': int(row['s']),
            'time': float(row['ideal_parallel_time_s']),
            'mem': float(row['max_worker_memory_per_cpu_mb']),
        } for row in csv.DictReader(f)]

    r_vals = [10, 11, 12, 13]
    colors = {10: '#1f77b4', 11: '#ff7f0e', 12: '#2ca02c', 13: '#d62728'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))

    for r_val in r_vals:
        c = colors[r_val]
        kd = sorted([d for d in kira_data if d['r'] == r_val], key=lambda x: x['s'])
        md = sorted([d for d in ml_data   if d['r'] == r_val], key=lambda x: x['s'])

        ks, kt, km = [d['s'] for d in kd], [d['time'] for d in kd], [d['mem'] for d in kd]
        ms, mt, mm = [d['s'] for d in md], [d['time'] for d in md], [d['mem'] for d in md]

        ax1.plot(ks, kt, color=c, linewidth=2, marker='o', markersize=6, linestyle=(0, (4, 3)))
        ax1.plot(ms, mt, '-', color=c, linewidth=2, marker='s', markersize=6, label=f'$r = {r_val}$')
        ax2.plot(ks, km, color=c, linewidth=2, marker='o', markersize=6, linestyle=(0, (4, 3)))
        ax2.plot(ms, mm, '-', color=c, linewidth=2, marker='s', markersize=6, label=f'$r = {r_val}$')

    ax1.plot([], [], color='gray', linewidth=2, marker='o', markersize=6, linestyle=(0, (4, 3)), label='Kira')
    ax1.plot([], [], '-', color='gray', linewidth=2, marker='s', markersize=6, label='SAILIR')

    ax1.set_xlabel('$s$', fontsize=13); ax1.set_ylabel('Time (s)', fontsize=13)
    ax1.set_yscale('log'); ax1.set_xticks([4, 5, 6, 7]); ax1.grid(True, alpha=0.3, which='both')
    ax2.set_xlabel('$s$', fontsize=13); ax2.set_ylabel('Peak memory (MB)', fontsize=13)
    ax2.set_yscale('log'); ax2.set_xticks([4, 5, 6, 7]); ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout(); plt.subplots_adjust(bottom=0.22)
    fig.legend(*ax1.get_legend_handles_labels(), fontsize=10, ncol=6,
               handlelength=3.5, loc='lower center', bbox_to_anchor=(0.5, 0.0))
    plt.savefig(args.out, bbox_inches='tight')
    print(f'Saved to {args.out}')

if __name__ == '__main__':
    main()
