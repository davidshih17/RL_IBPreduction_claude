"""Small utility functions shared by the SAILIR beam-search / orchestrator scripts.

Provides:
    get_sector_mask    -- N_DENOMINATORS-bit sector mask from an integral index tuple
    filter_to_sector   -- restrict an expression to one sector
    get_non_masters    -- non-master integrals in a given sector (default top)
    max_weight         -- max (r, s) weight among non-masters

TOPOLOGY-AGNOSTIC since C5: sector mask length is taken from
sailir.ibp_env.N_DENOMINATORS (set by init_from_topology). Callers
must call ibp_env.init_from_topology(t) before using these helpers.
"""

import sys
from pathlib import Path

# Make the sailir/ package importable from this scripts/eval/ directory.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir import ibp_env
from sailir.ibp_env import filter_top_sector, is_master, weight


def get_sector_mask(integral):
    """Return the sector mask (list of 0/1, length N_DENOMINATORS) for an integral.

    Walks denominator positions in order (skipping ISP positions, which by
    convention come after the denominator positions in our topologies).
    Matches the convention used by sailir.ibp_env.get_sector.
    """
    n_indices = ibp_env.N_INDICES
    isp = set(ibp_env.ISP_POSITIONS)
    out = []
    for i in range(n_indices):
        if i in isp:
            continue
        out.append(1 if integral[i] > 0 else 0)
    return out


def filter_to_sector(expr, sector):
    """Restrict an expression dict to integrals whose sector matches ``sector``."""
    return {k: v for k, v in expr.items() if get_sector_mask(k) == list(sector)}


def get_non_masters(expr, target_sector=None):
    """Return the non-master integrals from ``target_sector`` (or the top sector if None)."""
    if target_sector is None:
        filtered = filter_top_sector(expr)
    else:
        filtered = filter_to_sector(expr, target_sector)
    return {k: v for k, v in filtered.items() if not is_master(k)}


def max_weight(expr, target_sector=None):
    """Return the maximum (r, s) weight among non-masters in ``expr``."""
    nms = get_non_masters(expr, target_sector)
    if not nms:
        return (0, 0)
    return max((weight(k)[0], weight(k)[1]) for k in nms)
