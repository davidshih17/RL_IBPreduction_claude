#!/usr/bin/env python
"""For each step-166 baseline survivor, look forward to step N and count
how many descendants it has in the step-N beam (matched by path prefix).
This gives us "winners" (many descendants) and "losers" (few/none).

Then compare per-survivor features of their RS:
  - len(RS), total resolved-value size
  - n_useful_K (K in RS with sol_K ∩ expr_nm != empty)
  - max_w distribution of K's
  - score
  - cumulative -log(p) trajectory
to see if anything correlates with descendant count.
"""
import argparse
import pickle
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    weight,
)
from beam_search_utils import get_non_masters


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_dir')
    p.add_argument('topology')
    p.add_argument('--start-step', type=int, default=166)
    p.add_argument('--end-steps', type=str, default='180,200,220,240,260')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)

    with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step{args.start_step:04d}', 'rb') as f:
        c_start = pickle.load(f)
    ts = tuple(c_start['target_sector'])

    print(f'Start step {args.start_step}: {len(c_start["beam"])} survivors')
    start_paths = [pk(s) for s in c_start['beam']]

    # For each end step, count descendants per start ancestor.
    end_steps = [int(x) for x in args.end_steps.split(',')]
    desc_counts_by_step = {}
    for end_step in end_steps:
        try:
            with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step{end_step:04d}', 'rb') as f:
                ce = pickle.load(f)
        except FileNotFoundError:
            print(f'  step {end_step} ckpt missing, skip')
            continue
        # Each descendant's path starts with one of the start_paths
        counts = Counter()
        for sc in ce['beam']:
            cp = pk(sc)
            for i, sp in enumerate(start_paths):
                if cp[:len(sp)] == sp:
                    counts[i] += 1
                    break
        desc_counts_by_step[end_step] = counts
        n_winners = sum(1 for v in counts.values() if v > 0)
        n_max_desc = max(counts.values()) if counts else 0
        print(f'  step {end_step}: {n_winners}/{len(start_paths)} start-ancestors '
              f'have descendants; max desc count = {n_max_desc}')

    # Compute features for each start survivor
    print(f'\nComputing features for {len(c_start["beam"])} start survivors...')
    feats = []
    for i, s in enumerate(c_start['beam']):
        nm = get_non_masters(s['expr'], ts)
        expr_nm_set = set(nm.keys())
        rs = s['resolved_subs']
        n_rs = len(rs)
        total_rs_value_size = sum(len(v) for v in rs.values())
        # n_useful_K: K's whose resolved sol contains at least one current expr_nm element
        useful_K = [k for k, sol in rs.items() if any(t in expr_nm_set for t in sol)]
        n_useful = len(useful_K)
        # weight stats on K's in RS
        K_weights = [weight(k) for k in rs]
        K_w_max = max((w[0] for w in K_weights), default=0)
        K_w_total = sum(w[0] for w in K_weights)
        feats.append({
            'i': i,
            'score': float(s['score']),
            'path_len': len(s['path']),
            'n_nm': len(nm),
            'n_rs': n_rs,
            'rs_value_size': total_rs_value_size,
            'n_useful_K': n_useful,
            'K_w_max': K_w_max,
            'K_w_total': K_w_total,
        })

    # Correlate each feature with descendant count at each end step
    print('\nCorrelation (Pearson r) between start-survivor features and '
          'descendant count at each end-step:')
    keys = ('score', 'n_rs', 'rs_value_size', 'n_useful_K', 'K_w_max', 'K_w_total')
    for end_step, counts in desc_counts_by_step.items():
        nd = np.array([counts.get(f['i'], 0) for f in feats], dtype=float)
        print(f'  step {end_step} (desc counts: min={int(nd.min())} '
              f'max={int(nd.max())} mean={nd.mean():.2f}):')
        for k in keys:
            vals = np.array([f[k] for f in feats], dtype=float)
            if vals.std() > 0 and nd.std() > 0:
                r = np.corrcoef(vals, nd)[0, 1]
                print(f'    r(desc, {k:>16s}) = {r:+.3f}')
            else:
                print(f'    {k}: zero variance, skip')

    # Top-10 winners vs bottom-10 by descendants at the latest available end-step
    if desc_counts_by_step:
        last_step = max(desc_counts_by_step)
        counts = desc_counts_by_step[last_step]
        rank = sorted(feats, key=lambda f: -counts.get(f['i'], 0))
        print(f'\nTop survivors at step {args.start_step} by descendant count at step {last_step}:')
        for f in rank[:10]:
            print(f'  i={f["i"]:>2}  desc={counts.get(f["i"], 0):>3}  '
                  f'score={f["score"]:+.4f}  n_rs={f["n_rs"]:>3}  '
                  f'useful_K={f["n_useful_K"]:>3}  K_w_max={f["K_w_max"]:>2}  '
                  f'rs_value_sz={f["rs_value_size"]:>5}')
        print(f'Bottom (no descendants):')
        for f in rank:
            if counts.get(f['i'], 0) == 0:
                print(f'  i={f["i"]:>2}  desc=0  score={f["score"]:+.4f}  '
                      f'n_rs={f["n_rs"]:>3}  useful_K={f["n_useful_K"]:>3}  '
                      f'K_w_max={f["K_w_max"]:>2}  rs_value_sz={f["rs_value_size"]:>5}')


if __name__ == '__main__':
    sys.exit(main())
