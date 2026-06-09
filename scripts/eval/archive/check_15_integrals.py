"""For each of the 15 NEVER candidates, dump the full breakdown of why
the real is_master() classifies it."""
import sys
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')

from sailir import ibp_env
from sailir.ibp_env import (
    init_from_topology, set_prime, set_paper_masters_only,
    is_master, is_corner_integral, get_sector,
)
from sailir.topology import Topology


TOPO_DIR = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'

INTEGRALS = [
    (1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 0),
    (0, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0),
    (-1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    (1, -1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    (1, 1, 1, 0, 1, 1, 1, 1, -1, 0, 0),
    (1, 1, 1, -1, 1, 1, 1, 1, 0, 0, 0),
    (1, 1, -1, 1, 1, 1, 1, 1, 0, 0, 0),
    (1, 1, 1, 1, -1, 0, 1, 1, 0, 0, 0),
    (1, 1, 1, 0, -1, 1, 1, 1, 0, 0, 0),
    (-1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0),
    (-1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0),
    (1, 0, 1, 0, -1, 1, 1, 1, 0, 0, 0),
    (-1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0),
    (-1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0),
]


def main():
    topo = Topology.from_dir(TOPO_DIR)
    set_prime(1009)
    set_paper_masters_only(False)
    init_from_topology(topo)

    master_sectors = {get_sector(m) for m in ibp_env.MASTERS_SET}
    print(f'family name        : {ibp_env.FAMILY_NAME}')
    print(f'MASTERS_SET size   : {len(ibp_env.MASTERS_SET)}')
    print(f'master sectors     : {len(master_sectors)}')
    print()
    print(f'{"in_MS":>6s} {"corner":>7s} {"sec_master":>11s} {"is_master":>10s}'
          f'  integral')
    for ig in INTEGRALS:
        in_ms = ig in ibp_env.MASTERS_SET
        corn = is_corner_integral(ig)
        sec = get_sector(ig)
        sec_in_ms = sec in master_sectors
        im = is_master(ig)
        print(f'{str(in_ms):>6s} {str(corn):>7s} {str(sec_in_ms):>11s} '
              f'{str(im):>10s}  I{list(ig)}')


if __name__ == '__main__':
    main()
