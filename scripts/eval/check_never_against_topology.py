"""Load the pentagonbox topology and check each NEVER integral against the
real is_master / MASTERS_SET. No more speculation — just the truth.
"""
import os, pickle, sys

# wire up paths to import sailir
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')

from sailir import ibp_env
from sailir.ibp_env import (
    init_from_topology, set_prime, set_paper_masters_only,
    is_master, is_corner_integral, get_sector, MASTERS_SET,
)
from sailir.topology import Topology


TOPO_DIR = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'
STATE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v6/replay_state.pkl'


def Lrs(ig):
    L = sum(1 for x in ig[:8] if x > 0)
    r = sum(x for x in ig if x > 0)
    s = -sum(x for x in ig if x < 0)
    return L, r, s


def main():
    # Match orchestrator: --no-paper-masters-only
    topo = Topology.from_dir(TOPO_DIR)
    set_prime(1009)
    set_paper_masters_only(False)
    init_from_topology(topo)
    print(f'topology loaded: {topo.family_name}')
    print(f'MASTERS_SET size: {len(ibp_env.MASTERS_SET)}')
    print()

    state = pickle.load(open(STATE, 'rb'))
    expr = state['active_expr']
    cache = state['cache']
    log_set = state['log_integrals']

    # Recompute NEVER using REAL is_master
    never = []
    n_real_master_in_expr = 0
    for ig in expr:
        if is_master(ig):
            n_real_master_in_expr += 1
            continue
        if ig in cache:
            continue
        if ig in log_set:
            continue
        never.append(ig)

    print(f'active_expr terms     : {len(expr)}')
    print(f'  real masters        : {n_real_master_in_expr}')
    print(f'  REAL NEVER count    : {len(never)}')
    print()
    print('=== each NEVER integral, with is_master / sector / corner check ===')
    print(f'{"L":>2s} {"r":>3s} {"s":>3s}  '
          f'{"is_master":>9s} {"in_MS":>5s} {"corner":>6s} {"sec_in_MS":>10s}'
          f'  integral')
    for ig in sorted(never, key=lambda i: (-Lrs(i)[0], -Lrs(i)[1], -Lrs(i)[2])):
        L, r, s = Lrs(ig)
        im = is_master(ig)
        in_ms = ig in ibp_env.MASTERS_SET
        corn = is_corner_integral(ig)
        sec = get_sector(ig)
        sec_in_master_sec = sec in {get_sector(m) for m in ibp_env.MASTERS_SET}
        print(f'{L:>2d} {r:>3d} {s:>3d}  '
              f'{str(im):>9s} {str(in_ms):>5s} {str(corn):>6s} '
              f'{str(sec_in_master_sec):>10s}  I{list(ig)}')


if __name__ == '__main__':
    main()
