#!/usr/bin/env python3
"""Load the versioned keep_step beam snapshots from probe_stuck_74_ckpt and, for
each step, print the best beam-state's top non-master integrals (by weight). This
shows whether the recurring best mw=(7,4) nm=1 is the SAME integral every step
(stuck re-deriving one state) or DIFFERENT (7,4) integrals (genuine exploration
that never drains). Run AFTER the probe has written some .keep_step files.
"""
import glob
import pickle
import sys
from pathlib import Path

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + '/scripts/eval')
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_prime,
                            set_paper_masters_only, weight, is_master)
from beam_search_utils import get_sector_mask

D = f'{BASE}/results/probe_stuck_74_ckpt'
START = (-4, 2, 1, 0, 1, 1, 1, 1, 0, 0, 0)

topo = Topology.from_dir(f'{BASE}/topology_input/pentagonbox')
init_from_topology(topo)
set_prime(1009)
set_paper_masters_only(False)            # round3 used --no-paper-masters-only
target_sector = tuple(get_sector_mask(START))


def load_ckpt(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
        states = [pickle.load(f) for _ in range(meta['n_states'])]
    return meta, states


files = sorted(glob.glob(f'{D}/ckpt.pkl.keep_step*'),
               key=lambda p: int(p.rsplit('keep_step', 1)[-1]))
if not files:
    print(f'no keep_step snapshots in {D} yet', flush=True)
    sys.exit(1)

print(f'{len(files)} snapshots found\n', flush=True)
hist_top74 = []          # the heaviest non-master integral at each step
for path in files:
    meta, states = load_ckpt(path)
    best = states[0]                      # beam is sorted best-first
    expr = best['expr']
    nms = [(ig, weight(ig)) for ig, c in expr.items()
           if c != 0 and not is_master(ig)]
    nms.sort(key=lambda x: (-x[1][0], -x[1][1], x[0]))
    top = nms[:3]
    heaviest = top[0][0] if top else None
    hist_top74.append((meta['step'], heaviest))
    tops = '  '.join(f'(w={w[0]},{w[1]})I{list(ig)}' for ig, w in top)
    print(f"step {meta['step']:4d} | path={len(best['path']):4d} | "
          f"n_nm={len(nms):2d} | top: {tops}", flush=True)

# Summary: is the heaviest non-master the same integral across all steps?
distinct = {h for _, h in hist_top74 if h is not None}
print(f'\n=== distinct heaviest non-master integrals across {len(hist_top74)} '
      f'snapshots: {len(distinct)} ===', flush=True)
for ig in sorted(distinct):
    steps = [s for s, h in hist_top74 if h == ig]
    print(f'  I{list(ig)}  weight={weight(ig)[:2]}  at {len(steps)} steps '
          f'(e.g. {steps[:6]})', flush=True)
