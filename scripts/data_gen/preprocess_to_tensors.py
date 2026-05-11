#!/usr/bin/env python3
"""Pack JSONL training data from generate_multisector_data.py into PyTorch tensors.

generate_multisector_data.py writes one JSON sample per line, with variable-
length expression, substitution, and action lists.  This script reads the
JSONL files, splits into train/val, and saves packed tensor dictionaries
(``train.pt``, ``val.pt``) that train_classifier.py loads directly.

Schema produced (consumed by scripts/train/train_classifier.py):
    expr_integrals (N_tokens, 7) long
    expr_coeffs    (N_tokens,)   long
    expr_offsets   (N+1,)        long  -- cumulative
    sub_integrals  (M_tokens, 7) long  -- substitution KEY integrals only
    sub_offsets    (N+1,)        long
    subs_raw       Python list[N]      -- full [[key, [[repl_i, coeff_i], ...]], ...]
    action_ibp_ops (A_tokens,)   long
    action_deltas  (A_tokens, 7) long
    action_offsets (N+1,)        long
    sector_masks   (N, 6)        long
    target_integrals (N, 7)      long
    labels         (N,)          long  -- oracle action index into each sample's
                                          action window

Usage:
    python scripts/data_gen/preprocess_to_tensors.py \\
        --input  data/raw_jsonl/             \\
        --output data/multisector/           \\
        --val-fraction 0.1                   \\
        --seed 0
"""

import argparse
import json
import random
from pathlib import Path

import torch


def iter_jsonl(input_path: Path):
    """Yield parsed samples from a single .jsonl file or all .jsonl files in a directory."""
    if input_path.is_dir():
        files = sorted(input_path.glob('*.jsonl')) + sorted(input_path.glob('*.jsonl.gz'))
    else:
        files = [input_path]

    for f in files:
        if f.suffix == '.gz':
            import gzip
            opener = lambda p: gzip.open(p, 'rt')
        else:
            opener = lambda p: open(p, 'r')
        with opener(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def pack(samples):
    """Pack a list of JSONL sample dicts into the tensor dictionary."""
    expr_integrals, expr_coeffs = [], []
    expr_offsets = [0]
    sub_integrals = []
    sub_offsets = [0]
    subs_raw = []
    action_ibp_ops, action_deltas = [], []
    action_offsets = [0]
    sector_masks, target_integrals, labels = [], [], []

    for s in samples:
        # Expression: list of [integral, coeff]
        for integral, coeff in s['expr']:
            expr_integrals.append(integral)
            expr_coeffs.append(int(coeff))
        expr_offsets.append(len(expr_integrals))

        # Substitutions: list of [key, [[repl_i, coeff_i], ...]]
        for key, _repls in s['subs']:
            sub_integrals.append(key)
        sub_offsets.append(len(sub_integrals))
        # subs_raw keeps the FULL structure (coeffs as Python ints, integrals as
        # plain lists) -- the collate_fn pads it on the fly.
        subs_raw.append([
            [list(key), [[list(ri), int(rc)] for ri, rc in repls]]
            for key, repls in s['subs']
        ])

        # Valid actions: list of [ibp_op, delta]
        for ibp_op, delta in s['valid_actions']:
            action_ibp_ops.append(int(ibp_op))
            action_deltas.append(delta)
        action_offsets.append(len(action_ibp_ops))

        sector_masks.append(s['sector_mask'])
        target_integrals.append(s['target'])
        labels.append(int(s['chosen_action_idx']))

    return {
        'expr_integrals':   torch.tensor(expr_integrals,  dtype=torch.long),
        'expr_coeffs':      torch.tensor(expr_coeffs,     dtype=torch.long),
        'expr_offsets':     torch.tensor(expr_offsets,    dtype=torch.long),
        'sub_integrals':    torch.tensor(sub_integrals,   dtype=torch.long) if sub_integrals
                            else torch.zeros((0, 7), dtype=torch.long),
        'sub_offsets':      torch.tensor(sub_offsets,     dtype=torch.long),
        'subs_raw':         subs_raw,
        'action_ibp_ops':   torch.tensor(action_ibp_ops,  dtype=torch.long),
        'action_deltas':    torch.tensor(action_deltas,   dtype=torch.long),
        'action_offsets':   torch.tensor(action_offsets,  dtype=torch.long),
        'sector_masks':     torch.tensor(sector_masks,    dtype=torch.long),
        'target_integrals': torch.tensor(target_integrals, dtype=torch.long),
        'labels':           torch.tensor(labels,          dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True,
                    help='JSONL file, or directory containing one or more *.jsonl(.gz) files')
    ap.add_argument('--output', required=True,
                    help='Output directory (will hold train.pt and val.pt)')
    ap.add_argument('--val-fraction', type=float, default=0.1,
                    help='Fraction of samples reserved for validation')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f'Reading from {args.input} ...', flush=True)
    all_samples = list(iter_jsonl(Path(args.input)))
    rng.shuffle(all_samples)
    n_total = len(all_samples)
    n_val = int(n_total * args.val_fraction)
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f'  total = {n_total}   train = {len(train_samples)}   val = {len(val_samples)}',
          flush=True)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Packing train ...', flush=True)
    torch.save(pack(train_samples), outdir / 'train.pt')
    print('Packing val ...', flush=True)
    torch.save(pack(val_samples),   outdir / 'val.pt')
    print(f'Wrote {outdir/"train.pt"} and {outdir/"val.pt"}', flush=True)


if __name__ == '__main__':
    main()
