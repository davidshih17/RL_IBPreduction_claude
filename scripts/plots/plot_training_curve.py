#!/usr/bin/env python3
"""Plot training and validation loss/top-1 accuracy vs epoch from train_v5.log.

Reproduces Fig. 2 of the paper.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).parent.parent.parent.resolve()

def parse_log(log_path):
    epochs, train_loss, val_loss, train_top1, val_top1 = [], [], [], [], []
    with open(log_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m = re.match(r'^Epoch (\d+)/\d+', lines[i])
        if m:
            epoch = int(m.group(1))
            i += 1; tm = re.match(r'^\s+Train: loss=([\d.]+), top1=([\d.]+)', lines[i])
            i += 1; vm = re.match(r'^\s+Val:\s+loss=([\d.]+), top1=([\d.]+)', lines[i])
            if tm and vm:
                epochs.append(epoch)
                train_loss.append(float(tm.group(1))); train_top1.append(float(tm.group(2)))
                val_loss.append(float(vm.group(1)));   val_top1.append(float(vm.group(2)))
        i += 1
    return epochs, train_loss, val_loss, train_top1, val_top1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',         default=str(REPO_DIR / 'results/training_log/train.log'))
    ap.add_argument('--out',         default=str(REPO_DIR / 'results/training_curve.pdf'))
    ap.add_argument('--best-epoch',  type=int, default=22)
    args = ap.parse_args()

    epochs, train_loss, val_loss, train_top1, val_top1 = parse_log(args.log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, '-', color='#1f77b4', linewidth=2.5, label='Train')
    ax1.plot(epochs, val_loss,   '-', color='#d62728', linewidth=2.5, label='Validation')
    ax1.axvline(x=args.best_epoch, color='gray', linestyle='--', alpha=0.5,
                label=f'Best checkpoint (epoch {args.best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Cross-entropy loss', fontsize=12)
    ax1.legend(fontsize=10); ax1.set_xlim(0, 31)

    ax2.plot(epochs, [100*x for x in train_top1], '-', color='#1f77b4', linewidth=2.5, label='Train')
    ax2.plot(epochs, [100*x for x in val_top1],   '-', color='#d62728', linewidth=2.5, label='Validation')
    ax2.axvline(x=args.best_epoch, color='gray', linestyle='--', alpha=0.5,
                label=f'Best checkpoint (epoch {args.best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12); ax2.set_ylabel('Top-1 accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10, loc='lower right'); ax2.set_xlim(0, 31)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')
    print(f'Saved to {args.out}')

if __name__ == '__main__':
    main()
