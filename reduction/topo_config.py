#!/usr/bin/env python
"""Topology selection for the reduction/routing layer.

SAILIR_TOPOLOGY environment variable keys everything; DEFAULT = pentagonbox so
all existing pentagonbox behavior is bit-identical with the flag unset. Set it
identically for the orchestrator AND all workers of a run (it rides the condor
environment= line next to SAILIR_SECTOR_RANK).

Each entry:
  N_IND / N_DEN        index slots / denominator slots (sector-mask bits)
  TOPO_DIR             Topology.from_dir input
  CANON_PKL            canonical sectors {rep_of, canonical} (sector-senior rank)
  CANON_MAPS_PKL       composite non-canonical -> canonical corner maps cache
  CANONICALIZE_MOD     module providing _transforms/image_unsigned/_reduce/P
                       (pentagonbox: momentum-engine 'canonicalize';
                        gravity3L:  store-backed 'canonicalize_GR')
"""
import os

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
TOPOLOGY = os.environ.get('SAILIR_TOPOLOGY', 'pentagonbox')

_CFG = {
    'pentagonbox': dict(
        N_IND=11, N_DEN=8,
        TOPO_DIR=os.path.join(ROOT, "topology_input/pentagonbox"),
        CANON_PKL=os.path.join(ROOT, "results/canonical_sectors_tkey.pkl"),
        CANON_MAPS_PKL=os.path.join(ROOT, "results/sector_canon_maps.pkl"),
        CANONICALIZE_MOD='canonicalize',
    ),
    'gravity3L': dict(
        N_IND=15, N_DEN=10,
        TOPO_DIR=os.path.join(ROOT, "topology_input/gravity3L"),
        # v2 = clean-den convention (298 canonical sectors; the ing-based v1
        # pkl over-merged 6 sectors via flip maps — see canonicalize_GR.py)
        CANON_PKL=os.path.join(ROOT, "results/canonical_sectors_GR_v2.pkl"),
        CANON_MAPS_PKL=os.path.join(ROOT, "results/sector_canon_maps_GR.pkl"),
        CANONICALIZE_MOD='canonicalize_GR',
    ),
}
if TOPOLOGY not in _CFG:
    raise RuntimeError(f"SAILIR_TOPOLOGY={TOPOLOGY!r} unknown (have {sorted(_CFG)})")

_c = _CFG[TOPOLOGY]
N_IND = _c['N_IND']
N_DEN = _c['N_DEN']
TOPO_DIR = _c['TOPO_DIR']
CANON_PKL = _c['CANON_PKL']
CANON_MAPS_PKL = _c['CANON_MAPS_PKL']
CANONICALIZE_MOD = _c['CANONICALIZE_MOD']


def canonicalize_module():
    """Import and return the topology's canonicalize provider module."""
    import importlib
    return importlib.import_module(CANONICALIZE_MOD)
