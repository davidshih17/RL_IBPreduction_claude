"""Print ALL non-master integrals in the (greedy beam=1) expr at run-steps
1 / 25 / 50 for v7 and v8 on probe 74. Needs the env for is_master."""
import os
import sys
import pickle
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(__file__))  # beam_search_utils

from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_paper_masters_only, weight
from beam_search_utils import get_non_masters, get_sector_mask

TOPO = 'topology_input/pentagonbox'
START = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)

init_from_topology(Topology.from_dir(TOPO))
set_paper_masters_only(False)   # matches --no-paper-masters-only
target_sector = tuple(get_sector_mask(START))


def load_expr(tag, run_step):
    p = f'results/traj_{tag}_74/picks/picks_step{run_step-1}.pkl'
    with open(p, 'rb') as f:
        task = pickle.load(f)[0]
    return dict(task[0])   # {integral: coeff}


def tkey(i):
    w = weight(i)
    return (-w[0], -w[1], w[2])


for run_step in (1, 25, 50):
    print("=" * 78)
    print(f"RUN-STEP {run_step}")
    print("=" * 78)
    for tag in ('v7', 'v8'):
        expr = load_expr(tag, run_step)
        nm = get_non_masters(expr, target_sector)
        print(f"  {tag}: {len(nm)} non-masters in target sector:")
        for integ in sorted(nm, key=tkey):
            w = weight(integ)
            print(f"      {list(integ)}   (r,s)=({w[0]},{w[1]})  |abs|={w[2]}  "
                  f"coeff={nm[integ]}")
        print()
