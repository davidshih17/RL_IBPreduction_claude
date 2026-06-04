#!/usr/bin/env python
"""For each step's survivors, list weight distribution of non-masters in
expr (target-sector projection). Also show the starting integral's weight
for comparison.
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
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment, weight,
)
from beam_search_utils import get_non_masters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_dir')
    p.add_argument('topology')
    p.add_argument('--steps', default='0,50,100,166,200,260')
    p.add_argument('--integral', default='-1,2,1,0,1,2,1,1,-3,0,0',
                   help='Starting integral (comma-separated)')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)

    start_int = tuple(int(x) for x in args.integral.split(','))
    print(f'Starting integral: {start_int}')
    print(f'Starting integral weight: {weight(start_int)}')
    print()

    for step_str in args.steps.split(','):
        step = int(step_str)
        try:
            with open(f'{args.ckpt_dir}/result.pkl.ckpt.r1.step{step:04d}', 'rb') as f:
                c = pickle.load(f)
        except FileNotFoundError:
            print(f'step {step}: ckpt missing')
            continue
        ts = tuple(c['target_sector'])
        s = c['beam'][0]
        nm = get_non_masters(s['expr'], ts)
        # weight (w1, w2) per non-master
        weights_nm = [weight(k) for k in nm]
        w_counter = Counter(weights_nm)
        max_w = max(weights_nm) if weights_nm else None
        below_start = sum(1 for w in weights_nm if w < weight(start_int))
        print(f'step {step:>3}: beam[0] n_non_masters={len(nm)} max_w={max_w}  '
              f'#nm with weight < start_integral_weight = {below_start}')
        for w, c_ in sorted(w_counter.items(), reverse=True):
            print(f'    weight={w}  count={c_}')
        # Also: any non-master that's already a "sub-sector" of an earlier sector?
        # Compare each nm's sector vs target_sector
        print()


if __name__ == '__main__':
    sys.exit(main())
