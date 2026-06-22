#!/usr/bin/env python
"""Load a thick checkpoint and advance ONE step in BOTH v4-rescue and
baseline code paths. Print detailed comparison of the resulting beams.

Usage:
  single_step_diff.py <ckpt_v4> <ckpt_base> <topology_dir>

  ckpt_v4    a thick checkpoint of v4-rescue at step K
  ckpt_base  a thick checkpoint of baseline at step K
  topology   topology dir

Both checkpoints must be at the same step K with the same beam composition
(action paths must match through step K — if not, the comparison isn't
meaningful). Each is loaded as a DeltaState beam; we run ONE step of
production beam_search on each, then diff the resulting step-(K+1) beams.

Output:
  - candidate sets (op, delta, target) — same?
  - candidate scores — same?
  - top-40 survivors — same paths?
  - First (op, delta) that differs in score or selection
"""
import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
)


def load(path):
    """Load thick OR thin checkpoint. Thick has 'aux_flat'; thin has just paths."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_v4')
    p.add_argument('ckpt_base')
    p.add_argument('topology')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)

    d_v4 = load(args.ckpt_v4)
    d_base = load(args.ckpt_base)

    print(f'v4   step={d_v4["step"]} mode={d_v4.get("checkpoint_mode","?")} '
          f'beam_size={len(d_v4["beam"])}')
    print(f'base step={d_base["step"]} mode={d_base.get("checkpoint_mode","?")} '
          f'beam_size={len(d_base["beam"])}')

    if d_v4['step'] != d_base['step']:
        print('ERROR: checkpoint steps differ; cannot diff lockstep.')
        return 1

    # Match beams by sorted path; expect them to be equal if the runs were
    # bit-identical up to step K.
    def sig(s):
        return (tuple(s['max_w']), s['n_non_masters'],
                tuple(tuple(a) for a in s['path']))
    sigs_v4 = sorted(sig(s) for s in d_v4['beam'])
    sigs_base = sorted(sig(s) for s in d_base['beam'])
    n_match = sum(1 for a, b in zip(sigs_v4, sigs_base) if a == b)
    print(f'Beam path-signature match: {n_match}/{len(sigs_v4)}')

    if n_match < len(sigs_v4):
        # Find first divergence
        for i, (a, b) in enumerate(zip(sigs_v4, sigs_base)):
            if a != b:
                print(f'  first beam-sig diff at rank {i}:')
                print(f'    v4: max_w={a[0]} nm={a[1]} path[-3:]={a[2][-3:]}')
                print(f'    base: max_w={b[0]} nm={b[1]} path[-3:]={b[2][-3:]}')
                break

    # TODO: load both as DeltaState, run one production step on each
    # (P1+P2+P3+P4), then diff candidate sets and scores. Requires importing
    # the production beam_search and either patching it to stop after one
    # step or implementing a single-step variant. Stub for now.
    print('\nSingle-step advance not yet implemented \u2014 use this output')
    print('to confirm the two checkpoints are aligned, then add the')
    print('one-step advance harness on top of beam_search_delta.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
