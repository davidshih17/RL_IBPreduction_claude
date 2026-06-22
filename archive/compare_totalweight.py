#!/usr/bin/env python3
"""For the stuck (7,4) probe: classify every best-state non-master in each
keep_step snapshot by its TOTAL-WEIGHT position relative to the START integral.

_target_key(i) = (-r, -s, abs_tuple); SMALLER key = HEAVIER (higher in the total
ordering). SUCCESS_TOTAL succeeds iff EVERY non-master has _target_key > start_key
(strictly BELOW the start). So a non-master BLOCKS success iff _target_key <=
start_key. This script reports, per snapshot's best state:
  ABOVE  : _target_key <  start_key  (strictly heavier than start -> blocks)
  EQUAL  : _target_key == start_key  (the start integral itself -> blocks)
  BELOW  : _target_key >  start_key  (strictly lighter -> OK passenger)
and lists the distinct heaviest non-masters with their ABOVE/EQUAL/BELOW verdict.
"""
import glob
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + '/scripts/eval')
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (init_from_topology, set_prime,
                            set_paper_masters_only, weight, is_master)
from beam_search_utils import get_sector_mask

# Probe dir: argv[1] (absolute or under results/), default the original maxweight probe.
_arg = sys.argv[1] if len(sys.argv) > 1 else 'probe_stuck_74_ckpt'
D = _arg if _arg.startswith('/') else f'{BASE}/results/{_arg}'
START = (-4, 2, 1, 0, 1, 1, 1, 1, 0, 0, 0)

topo = Topology.from_dir(f'{BASE}/topology_input/pentagonbox')
init_from_topology(topo)
set_prime(1009)
set_paper_masters_only(False)
target_sector = tuple(get_sector_mask(START))


def tk(i):
    w = weight(i)
    return (-w[0], -w[1], w[2])


START_KEY = tk(START)
print(f'START I{list(START)}  weight={weight(START)[:2]}  target_key={START_KEY}\n',
      flush=True)


def verdict(i):
    k = tk(i)
    if k < START_KEY:
        return 'ABOVE'   # heavier than start -> blocks success
    if k == START_KEY:
        return 'EQUAL'   # the start itself -> blocks
    return 'BELOW'       # lighter -> acceptable passenger


def load(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
        states = [pickle.load(f) for _ in range(meta['n_states'])]
    return meta, states


files = sorted(glob.glob(f'{D}/ckpt.pkl.keep_step*'),
               key=lambda p: int(p.rsplit('keep_step', 1)[-1]))
if not files:
    print('no snapshots yet', flush=True)
    sys.exit(1)

distinct = {}     # heaviest non-master -> verdict
for path in files:
    meta, states = load(path)
    best = states[0]
    nms = [ig for ig, c in best['expr'].items() if c != 0 and not is_master(ig)]
    cnt = {'ABOVE': 0, 'EQUAL': 0, 'BELOW': 0}
    for ig in nms:
        cnt[verdict(ig)] += 1
    nms.sort(key=tk)                       # heaviest (smallest key) first
    h = nms[0] if nms else None
    if h is not None:
        distinct[h] = verdict(h)
    hv = verdict(h) if h is not None else '-'
    print(f"step {meta['step']:4d} | n_nm={len(nms):2d} "
          f"(ABOVE={cnt['ABOVE']} EQUAL={cnt['EQUAL']} BELOW={cnt['BELOW']}) | "
          f"heaviest={hv} (w={weight(h)[:2]}) I{list(h)}", flush=True)

print('\n=== distinct heaviest non-masters + verdict vs start ===', flush=True)
for ig, v in sorted(distinct.items(), key=lambda kv: tk(kv[0])):
    print(f'  {v:5s}  w={weight(ig)[:2]}  key={tk(ig)}  I{list(ig)}', flush=True)
nab = sum(1 for v in distinct.values() if v in ('ABOVE', 'EQUAL'))
print(f'\nheaviest non-master was ABOVE/EQUAL start in {nab}/{len(distinct)} '
      f'distinct cases (BELOW = real progress)', flush=True)
