#!/usr/bin/env python
"""Quick check: identify the 4 step-166 ancestors that have descendants
at step 180, and report their (useful_K, rs_value_sz, score) features.
Specifically check if they're the 4 with useful_K=119.
"""
import argparse
import pickle
import sys
from collections import Counter

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, weight,
)
from beam_search_utils import get_non_masters


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_dir')
    p.add_argument('topology')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)

    with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step0166', 'rb') as f:
        c166 = pickle.load(f)
    with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step0180', 'rb') as f:
        c180 = pickle.load(f)
    ts = tuple(c166['target_sector'])
    start_paths = [pk(s) for s in c166['beam']]

    counts = Counter()
    for sc in c180['beam']:
        cp = pk(sc)
        for i, sp in enumerate(start_paths):
            if cp[:len(sp)] == sp:
                counts[i] += 1
                break

    print(f'Step 180 descendant counts per step-166 ancestor:')
    print(f'{"idx":>4} {"desc":>4} {"score":>11} {"useful_K":>8} {"rs_val_sz":>10}')
    for i, s in enumerate(c166['beam']):
        nm = get_non_masters(s['expr'], ts)
        expr_nm_set = set(nm.keys())
        rs = s['resolved_subs']
        useful_K = sum(1 for k, sol in rs.items() if any(t in expr_nm_set for t in sol))
        rs_val_sz = sum(len(v) for v in rs.values())
        desc = counts.get(i, 0)
        marker = ' <-- HAS DESC' if desc > 0 else ('  (uK=119)' if useful_K == 119 else '')
        print(f'{i:>4} {desc:>4} {s["score"]:>+11.5f} {useful_K:>8} {rs_val_sz:>10}{marker}')


if __name__ == '__main__':
    sys.exit(main())
