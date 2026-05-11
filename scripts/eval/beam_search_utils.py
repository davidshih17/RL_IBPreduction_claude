"""Small utility functions shared by the SAILIR beam-search / orchestrator scripts.

Provides:
    get_sector_mask    -- 6-bit sector mask from an integral index tuple
    filter_to_sector   -- restrict an expression to one sector
    get_non_masters    -- non-master integrals in a given sector (default top)
    max_weight         -- max (r, s) weight among non-masters
"""

import sys
from pathlib import Path

# Make the sailir/ package importable from this scripts/eval/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'sailir'))

from ibp_env import filter_top_sector, is_master, weight


def get_sector_mask(integral):
    """Return the 6-bit sector mask (list of 0/1) for an integral."""
    return [1 if integral[i] > 0 else 0 for i in range(6)]


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
