"""
IBP Reduction Environment for end-to-end evaluation.

Provides functions to:
1. Load IBP/LI templates
2. Apply actions to expressions
3. Enumerate valid actions
4. Run full reductions using a model

TOPOLOGY-AGNOSTIC since C5 refactor: dimensions (n_indices, n_denominators,
ISP positions), family parser name, kinematic invariants, and the master
basis are pulled from a `sailir.topology.Topology` object via
`init_from_topology(t)`. Until that's called, the module has no useful state
and most operations will raise.

Supports configurable prime for modular arithmetic via set_prime(p).
Smaller primes like 1009 make coefficients more learnable.
"""

import re
from pathlib import Path
from typing import Optional

import torch

from sailir import topology as _topology_module
from sailir.topology import Topology

# Default prime - can be changed with set_prime()
PRIME = 2147483647

# ---------------------------------------------------------------------------
# Topology-driven globals. Set by init_from_topology(); do not touch directly.
# ---------------------------------------------------------------------------
_TOPOLOGY: Optional[Topology] = None
N_INDICES: int = 0                # length of integral tuple
N_DENOMINATORS: int = 0           # number of physical propagators (= bits in sector mask)
ISP_POSITIONS: tuple = ()         # 0-indexed positions that are ISPs (never in denom in basis)
FAMILY_NAME: str = ""             # e.g. "trianglebox" or "TA" — parser key
KINEMATIC_INVARIANTS: list = []   # symbol names from kinematics.yaml
KINEMATICS: dict = {}             # symbol -> int value (for finite-field eval)
IBP_TEMPLATES: list = []          # from topology.ibp_templates
LI_TEMPLATES: list = []           # from topology.li_templates
MASTERS_SET: frozenset = frozenset()       # set of master integral tuples
MASTERS_BY_SECTOR: dict = {}      # sector_id -> [master tuples]
MASTER_SECTORS: frozenset = frozenset()    # sector ids that have masters


def init_from_topology(topology: Topology) -> None:
    """Initialise module globals from a Topology.

    Must be called once at process startup, after `Topology.from_dir(...)`.
    Subsequent calls overwrite the previous topology.
    """
    global _TOPOLOGY, N_INDICES, N_DENOMINATORS, ISP_POSITIONS
    global FAMILY_NAME, KINEMATIC_INVARIANTS, KINEMATICS
    global IBP_TEMPLATES, LI_TEMPLATES
    global MASTERS_SET, MASTERS_BY_SECTOR, MASTER_SECTORS
    _TOPOLOGY = topology
    _topology_module.configure(topology)
    N_INDICES = topology.n_indices
    N_DENOMINATORS = topology.n_denominators
    ISP_POSITIONS = tuple(topology.isp_positions)
    FAMILY_NAME = topology.family_name
    KINEMATIC_INVARIANTS = list(topology.kinematic_invariants)
    KINEMATICS = dict(topology.kinematics_values)
    # IBP/LI templates are stored as { ibp_op_index: [(shift, coeff_str), ...] }
    # in this module's existing convention; convert from Topology's list form.
    IBP_TEMPLATES = {i: list(tpl) for i, tpl in enumerate(topology.ibp_templates)}
    LI_TEMPLATES = {i: list(tpl) for i, tpl in enumerate(topology.li_templates)}
    MASTERS_SET = frozenset(topology.masters)
    MASTERS_BY_SECTOR = {s: list(ms) for s, ms in topology.masters_by_sector.items()}
    MASTER_SECTORS = frozenset(topology.masters_by_sector.keys())


def set_prime(p):
    """Set the prime for modular arithmetic."""
    global PRIME
    PRIME = p
    print(f"IBP environment using PRIME = {PRIME}")


def set_kinematics(**kwargs):
    """Update specific kinematic invariant values.

    After init_from_topology(...), call e.g. set_kinematics(d=41, s12=31, ...).
    Keys must already be present in the topology's kinematics dict.
    """
    global KINEMATICS
    for k, v in kwargs.items():
        if k not in KINEMATICS:
            raise KeyError(
                f"set_kinematics: unknown symbol {k!r}; topology declares "
                f"{sorted(KINEMATICS.keys())}"
            )
        KINEMATICS[k] = v


# Default paths (kept for backward compat with old trianglebox scripts that
# read IBP/LI directly from scripts/data_gen/ rather than topology_input/).
IBP_PATH = Path(__file__).parent.parent / 'scripts/data_gen/IBP'
LI_PATH = Path(__file__).parent.parent / 'scripts/data_gen/LI'


def mod_inverse(a, p=None):
    """Compute modular inverse using extended Euclidean algorithm."""
    if p is None:
        p = PRIME
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    a = a % p
    if a < 0:
        a += p
    _, x, _ = extended_gcd(a, p)
    return x % p


def parse_templates(path, family_name: Optional[str] = None):
    """Parse IBP or LI template file.

    family_name defaults to the currently configured topology's family name
    (so legacy callers like `parse_templates(IBP_PATH)` continue to work).
    """
    if family_name is None:
        family_name = FAMILY_NAME
    if not family_name:
        raise RuntimeError(
            "parse_templates: no family_name available. Call "
            "ibp_env.init_from_topology(t) first, or pass family_name=..."
        )
    pat = re.compile(rf'{re.escape(family_name)}\[([^\]]+)\]\*\(([^)]+)\)')
    templates = {}
    current = None
    terms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current is not None and terms:
                    templates[current] = terms
                current = None
                terms = []
                continue
            if '\u2192' in line:  # the original "→" arrow
                line = line.split('\u2192', 1)[1]
            m = pat.match(line)
            if m:
                shift = tuple(int(x.strip()) for x in m.group(1).split(','))
                coeff_str = m.group(2)
                if current is None:
                    current = len(templates)
                terms.append((shift, coeff_str))
    if current is not None and terms:
        templates[current] = terms
    return templates


def eval_coeff(coeff_str, seed):
    """Evaluate a coefficient string at a given seed, mod PRIME.

    Binds a0..a{N-1} from `seed`, plus every kinematic invariant declared
    by the topology (e.g. 'd', 'm2', 'm3' for trianglebox; 'd', 's12', ...
    for pentagon-box). Returns 0 on eval failure (matching old behaviour).
    """
    ns = {f'a{i}': seed[i] for i in range(N_INDICES)}
    ns.update(KINEMATICS)  # d + invariant symbols -> integer values
    try:
        return eval(coeff_str.replace('^', '**'), {"__builtins__": {}}, ns) % PRIME
    except Exception:
        return 0


def get_raw_equation(ibp_t, li_t, ibp_op, seed):
    """Get raw IBP equation for given operator and seed.

    Action indices: 0..n_ibp-1 are IBP, n_ibp..n_ibp+n_li-1 are LI.
    """
    n_ibp = len(ibp_t)
    if ibp_op >= n_ibp:
        template = li_t.get(ibp_op - n_ibp, [])
    else:
        template = ibp_t.get(ibp_op, [])
    eq = {}
    for shift, coeff_str in template:
        c = eval_coeff(coeff_str, seed)
        if c != 0:
            integral = tuple(seed[i] + shift[i] for i in range(N_INDICES))
            eq[integral] = c
    return eq


def apply_substitution(expr, sub_int, sol):
    """Apply a substitution to an expression."""
    if sub_int not in expr:
        return expr
    coeff = expr[sub_int]
    new_expr = {k: v for k, v in expr.items() if k != sub_int}
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % PRIME
        if integral in new_expr:
            new_expr[integral] = (new_expr[integral] + new_coeff) % PRIME
        else:
            new_expr[integral] = new_coeff
    return {k: v for k, v in new_expr.items() if v != 0}


def apply_substitution_target_only(expr_t, sub_int, sol, target_sector):
    """Apply substitution, evolving ONLY the target-sector content. Sub-sector
    contributions from sol are DISCARDED here. The caller is responsible for
    reconstructing the full expression at the end by replaying the path of
    applied substitutions against the original start_expr.

    This is Option F's cheap-per-step path: no sub_accum to copy, no
    sub-sector dict ops at all. Per-step cost drops to O(|expr_t| + |sol|)
    with no |sub_accum| factor.

    Args:
        expr_t: target-sector-only expression dict
        sub_int: integral being substituted (must be target-sector)
        sol: replacement linear combination
        target_sector: tuple sector mask defining target-sector membership

    Returns:
        new_expr_t (new dict; caller's expr_t not mutated).
    """
    if sub_int not in expr_t:
        return expr_t
    coeff = expr_t[sub_int]
    new_expr_t = {k: v for k, v in expr_t.items() if k != sub_int}
    nd = N_DENOMINATORS
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % PRIME
        if new_coeff == 0:
            continue
        # Skip non-target-sector contributions — they're "passenger" terms
        # that will be reconstructed at the end by the caller.
        in_target = True
        for i in range(nd):
            if ((1 if integral[i] > 0 else 0) != target_sector[i]):
                in_target = False
                break
        if not in_target:
            continue
        if integral in new_expr_t:
            s = (new_expr_t[integral] + new_coeff) % PRIME
            if s == 0:
                del new_expr_t[integral]
            else:
                new_expr_t[integral] = s
        else:
            new_expr_t[integral] = new_coeff
    return new_expr_t


def apply_substitution_split(expr_t, sub_accum, sub_int, sol, target_sector):
    """Apply substitution, splitting result by sector.

    Like apply_substitution but separates target-sector terms from sub-sector
    terms. expr_t is the active target-sector expression; sub_accum holds the
    sub-sector "passenger" terms that don't affect search decisions but must
    be tracked for the final output. This saves O(|sub_accum|) per call by
    not iterating sub_accum at all (it's only added to, never read here).

    Args:
        expr_t: target-sector-only expression dict (modified to remove sub_int
                if present, and to receive any target-sector terms from sol)
        sub_accum: sub-sector accumulator dict (sub-sector terms from sol added)
        sub_int: the integral being substituted (must be in target sector)
        sol: replacement linear combination (mix of target/sub-sector)
        target_sector: tuple sector mask defining what counts as "target sector"

    Returns:
        (new_expr_t, new_sub_accum) — both NEW dicts (caller's not mutated).
    """
    if sub_int not in expr_t:
        return expr_t, sub_accum
    coeff = expr_t[sub_int]
    new_expr_t = {k: v for k, v in expr_t.items() if k != sub_int}
    new_sub_accum = dict(sub_accum)
    nd = N_DENOMINATORS
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % PRIME
        if new_coeff == 0:
            continue
        # Inline sector mask compare for hot-loop speed (avoid tuple alloc).
        in_target_sector = True
        for i in range(nd):
            if ((1 if integral[i] > 0 else 0) != target_sector[i]):
                in_target_sector = False
                break
        d = new_expr_t if in_target_sector else new_sub_accum
        if integral in d:
            s = (d[integral] + new_coeff) % PRIME
            if s == 0:
                del d[integral]
            else:
                d[integral] = s
        else:
            d[integral] = new_coeff
    return new_expr_t, new_sub_accum


def apply_all_substitutions(expr, subs):
    """Apply all substitutions until fixed point."""
    if not subs:
        return dict(expr)

    result = dict(expr)
    changed = True
    while changed:
        changed = False
        for sub_int, sol in subs.items():
            if sub_int in result and result[sub_int] != 0:
                coeff = result.pop(sub_int)
                for k, v in sol.items():
                    new_coeff = (coeff * v) % PRIME
                    if k in result:
                        result[k] = (result[k] + new_coeff) % PRIME
                    else:
                        result[k] = new_coeff
                changed = True
        result = {k: v for k, v in result.items() if v != 0}
    return result


def resolve_subs(subs):
    """Precompute fully-resolved substitutions.

    After resolution, each substitution's value contains NO keys from subs,
    so apply_resolved_subs can apply everything in a single pass.

    This is done once when subs changes, rather than iterating on every call.
    """
    if not subs:
        return {}

    # Deep copy the subs
    resolved = {k: dict(v) for k, v in subs.items()}

    # Keep resolving until no substitution value contains another substitution key
    changed = True
    while changed:
        changed = False
        for key in resolved:
            value = resolved[key]
            # Check if this value contains any keys from resolved
            keys_to_expand = [k for k in value if k in resolved]
            if keys_to_expand:
                # Expand those keys
                new_value = dict(value)
                for expand_key in keys_to_expand:
                    coeff = new_value.pop(expand_key)
                    for k, v in resolved[expand_key].items():
                        new_coeff = (coeff * v) % PRIME
                        if k in new_value:
                            new_value[k] = (new_value[k] + new_coeff) % PRIME
                        else:
                            new_value[k] = new_coeff
                # Clean zeros
                new_value = {k: v for k, v in new_value.items() if v != 0}
                if new_value != resolved[key]:
                    resolved[key] = new_value
                    changed = True

    return resolved


def apply_resolved_subs(expr, resolved_subs):
    """Apply pre-resolved substitutions in a single pass.

    Since resolved_subs values don't contain any resolved_subs keys,
    we only need ONE pass through the expression (no iteration to fixed point).
    """
    if not resolved_subs:
        return dict(expr)

    result = dict(expr)

    # Single pass: expand any keys that are in resolved_subs
    keys_to_expand = [k for k in result if k in resolved_subs]
    for sub_int in keys_to_expand:
        if sub_int in result and result[sub_int] != 0:
            coeff = result.pop(sub_int)
            for k, v in resolved_subs[sub_int].items():
                new_coeff = (coeff * v) % PRIME
                if k in result:
                    result[k] = (result[k] + new_coeff) % PRIME
                else:
                    result[k] = new_coeff

    return {k: v for k, v in result.items() if v != 0}


def add_sub_to_resolved(resolved_subs, target, sol):
    """Incrementally add a new substitution to resolved_subs.

    Instead of recomputing the full transitive closure, we:
    1. Resolve sol against existing resolved_subs (single pass)
    2. Add target -> resolved_sol
    3. Update any existing entries that contain target

    Returns new resolved_subs dict.
    """
    # Step 1: Resolve the new sol against existing resolved_subs
    resolved_sol = apply_resolved_subs(sol, resolved_subs)

    # Step 2: Create new dict with the new entry
    new_resolved = {k: dict(v) for k, v in resolved_subs.items()}
    new_resolved[target] = resolved_sol

    # Step 3: Update existing entries that contain target
    for key in resolved_subs:  # Only iterate over OLD keys
        value = new_resolved[key]
        if target in value:
            # Expand target in this value
            coeff = value.pop(target)
            for k, v in resolved_sol.items():
                new_coeff = (coeff * v) % PRIME
                if k in value:
                    value[k] = (value[k] + new_coeff) % PRIME
                else:
                    value[k] = new_coeff
            # Clean zeros
            new_resolved[key] = {k: v for k, v in value.items() if v != 0}

    return new_resolved


def solve_ibp_for(ibp, target):
    """Solve IBP equation for target integral."""
    if target not in ibp or ibp[target] == 0:
        return None
    neg_inv = (PRIME - mod_inverse(ibp[target])) % PRIME
    return {k: (neg_inv * v) % PRIME for k, v in ibp.items() if k != target}


def is_top_sector(integral):
    """Check if integral is in the topology's top sector (all denominators >= 1)."""
    for i in range(N_INDICES):
        if i in ISP_POSITIONS:
            continue
        if integral[i] < 1:
            return False
    return True


# Backwards-compat alias for code that still imports PAPER_MASTERS.
# After init_from_topology() this is the topology's master set.
def _paper_masters_view() -> frozenset:
    """Lazy proxy to MASTERS_SET (populated by init_from_topology)."""
    return MASTERS_SET


PAPER_MASTERS = _paper_masters_view  # callable returning the set, for explicit lookups
                                     # (legacy callers using `x in PAPER_MASTERS` should
                                     # now use `is_master(x)` or `x in MASTERS_SET`)


def get_sector(integral):
    """Sector tuple of length N_DENOMINATORS: 1 at each propagator slot with > 0."""
    # Iterate over denominator positions (skipping ISP positions).
    out = []
    for i in range(N_INDICES):
        if i in ISP_POSITIONS:
            continue
        out.append(1 if integral[i] > 0 else 0)
    return tuple(out)


def get_sector_id(integral):
    """Integer sector id (Kira convention: bit i for array index i if a_i > 0)."""
    return sum((1 << i) for i, a in enumerate(integral) if a > 0)


# Sectors covered by the topology's master basis (sector tuples, like get_sector output).
# Recomputed each access to handle re-init; cheap.
def _master_sector_tuples() -> frozenset:
    return frozenset(get_sector(m) for m in MASTERS_SET)


PAPER_MASTER_SECTORS = _master_sector_tuples

# Global flag to control whether to reduce to paper masters only
# If True, only the 16 paper masters are considered masters
# If False (default), corner integrals in uncovered sectors are also masters
PAPER_MASTERS_ONLY = False


def set_paper_masters_only(value):
    """Set whether to reduce to paper masters only (no corner integrals)."""
    global PAPER_MASTERS_ONLY
    PAPER_MASTERS_ONLY = value


def is_corner_integral(integral):
    """Check if integral is a corner integral.

    A corner integral has:
    - Every denominator position is 0 or 1 (no dots, no numerator)
    - Every ISP position is 0
    """
    for j in range(N_INDICES):
        if j in ISP_POSITIONS:
            if integral[j] != 0:
                return False
        else:
            if integral[j] not in (0, 1):
                return False
    return True


def is_master(i):
    """Check if integral is a master integral.

    Masters are:
    1. The topology's master basis (loaded via init_from_topology)
    2. Corner integrals in sectors not covered by the master basis
       (unless PAPER_MASTERS_ONLY is True)
    """
    if i in MASTERS_SET:
        return True
    if PAPER_MASTERS_ONLY:
        return False
    if get_sector(i) not in _master_sector_tuples():
        return is_corner_integral(i)
    return False


def is_higher_sector(i):
    """DEPRECATED: hardcoded for trianglebox sector 53 in legacy code.

    Raises NotImplementedError for any non-trianglebox topology. Use
    action_introduces_higher_sector_general() instead.
    """
    if FAMILY_NAME != "trianglebox":
        raise NotImplementedError(
            f"is_higher_sector is hardcoded for trianglebox sector 53; "
            f"current topology family is {FAMILY_NAME!r}. Use "
            f"action_introduces_higher_sector_general() instead."
        )
    return is_top_sector(i) and (i[1] >= 1 or i[3] >= 1 or i[6] >= 1)


def weight(i):
    return (sum(max(0, x) for x in i), -sum(min(0, x) for x in i), tuple(abs(x) for x in i))


def filter_top_sector(expr):
    """Filter to top sector (positions 0,2,4,5 >= 1). Includes higher sectors."""
    return {k: v for k, v in expr.items() if is_top_sector(k)}


def get_target(expr):
    """Get highest weight non-master integral."""
    top_only = filter_top_sector(expr)
    non_masters = {k: v for k, v in top_only.items() if not is_master(k)}
    if not non_masters:
        return None
    return max(non_masters.keys(),
               key=lambda x: (weight(x)[0], weight(x)[1], [-a for a in weight(x)[2]]))


def is_subsector(integral, target):
    """Check if integral's sector is a subsector of target's sector.

    A sector A is a subsector of B if every propagator that is on in A
    is also on in B. In other words, A can only have propagators where B has them.
    This includes the ISP (index 6).
    """
    for i in range(N_INDICES):  # Check all 7 indices including ISP
        if target[i] <= 0 and integral[i] > 0:
            return False
    return True


# Process-wide cache mapping integral tuple -> sector bitmask.
# Memoizes the (otherwise) per-call 11-element loop. Each entry is ~100 bytes
# (tuple key + small int + dict slot); the cache size is bounded by the
# number of unique integrals seen by this worker process (typically 100k-1M
# during a long reduction = 10-100 MB).
_BITMASK_CACHE = {}


def sector_bitmask(integral):
    """Compute a sector bitmask: bit i set iff integral[i] > 0.

    Memoized in `_BITMASK_CACHE` (process-wide). Used to vectorize
    is_subsector checks. Equivalent to:
        is_subsector(integral, target) == ((bitmask(integral) & ~bitmask(target)) == 0).
    """
    bm = _BITMASK_CACHE.get(integral)
    if bm is None:
        bm = 0
        for i in range(N_INDICES):
            if integral[i] > 0:
                bm |= (1 << i)
        _BITMASK_CACHE[integral] = bm
    return bm


def cached_union_bitmask(cached_eq):
    """OR of sector bitmasks across cached_eq.keys().

    By distributivity, action_introduces_outside_sector(cached_eq, target)
    becomes ((union_bm & ~target_bm) != 0) — one bitwise check instead of an
    iteration over every key. Stored once per indirect_cache entry to make
    the per-target filter check O(1). Uses _BITMASK_CACHE so repeated
    integrals across many indirect_cache entries are looked up, not
    recomputed.
    """
    u = 0
    cache = _BITMASK_CACHE  # local alias for inner-loop speed
    for k in cached_eq:
        bm = cache.get(k)
        if bm is None:
            bm = 0
            for i in range(N_INDICES):
                if k[i] > 0:
                    bm |= (1 << i)
            cache[k] = bm
        u |= bm
    return u


def action_introduces_outside_sector(cached_eq, target):
    """Check if an IBP equation introduces integrals outside target's sector hierarchy.

    Returns True if any integral is NOT a subsector of target (i.e., has propagators
    that target doesn't have - either lateral or higher sector).
    """
    for integral in cached_eq.keys():
        if not is_subsector(integral, target):
            return True
    return False


def integral_in_exact_sector(integral, target_sector):
    """Check if integral is EXACTLY in target_sector (not a subsector).

    Args:
        integral: 7-element tuple (6 propagators + 1 ISP)
        target_sector: 6-element list/tuple from get_sector_mask (just propagators)

    Returns:
        True if integral's sector mask exactly matches target_sector
    """
    for i in range(N_DENOMINATORS):
        integral_has_prop = integral[i] > 0
        target_has_prop = target_sector[i] == 1
        if integral_has_prop != target_has_prop:
            return False
    return True


def integral_in_sector_or_subsector(integral, target_sector):
    """Check if integral is in target_sector or a subsector of it.

    Args:
        integral: 7-element tuple (6 propagators + 1 ISP)
        target_sector: 6-element list/tuple from get_sector_mask (just propagators)

    Returns:
        True if integral's sector is target_sector or a subsector
    """
    # Check first 6 indices (propagators) - integral can only have propagators
    # where target_sector has them (subsector condition)
    for i in range(N_DENOMINATORS):
        # If target_sector has this propagator OFF (0) but integral has it ON (>0),
        # then integral is in a different/higher sector
        if target_sector[i] == 0 and integral[i] > 0:
            return False
    return True


def filter_subs_to_exact_sector(subs, target_sector):
    """Filter replacement terms WITHIN each sub to only exact target sector.

    When reducing sector S, replacement terms in lower sectors don't matter
    yet - they'll be dealt with when we reduce those sectors. This filters
    the replacement dict within each substitution to remove lower-sector terms.

    Example: If target_sector is [1,1,1,1,1,1] and a sub is:
        subs[(3,2,1,...)] = {
            (2,2,1,2,2,2,-5): 100,  # sector [1,1,1,1,1,1] - KEEP
            (0,2,1,2,2,2,-5): 50,   # sector [0,1,1,1,1,1] - DELETE
        }
    We keep only the first term in the replacement.

    Args:
        subs: dict mapping integral -> {replacement_integral: coeff, ...}
        target_sector: 6-element list from get_sector_mask (propagator mask)

    Returns:
        dict: subs with replacement terms filtered to exact sector only
    """
    if not subs or target_sector is None:
        return subs

    filtered = {}
    for source, replacement in subs.items():
        # Filter the replacement dict to only keep terms in exact target sector
        filtered_replacement = {}
        for term, coeff in replacement.items():
            if integral_in_exact_sector(term, target_sector):
                filtered_replacement[term] = coeff
        # Only include this sub if it has any remaining terms
        if filtered_replacement:
            filtered[source] = filtered_replacement

    return filtered


def filter_subs_by_sector(subs, target_sector):
    """Filter subs to only include entries relevant to target sector.

    DEPRECATED: Use filter_subs_to_exact_sector instead for better performance.
    This function keeps subsectors which doesn't reduce subs for top sector.
    """
    if not subs or target_sector is None:
        return subs

    filtered = {}
    for integral, replacement in subs.items():
        if integral_in_sector_or_subsector(integral, target_sector):
            filtered[integral] = replacement

    return filtered


def filter_resolved_subs_by_sector(resolved_subs, target_sector):
    """Filter resolved_subs to only include entries relevant to target sector.

    Same as filter_subs_by_sector but for the resolved form.
    """
    return filter_subs_by_sector(resolved_subs, target_sector)


def filter_resolved_subs_to_exact_sector(resolved_subs, target_sector):
    """Filter resolved_subs to only integrals EXACTLY in target sector.

    Same as filter_subs_to_exact_sector but for the resolved form.
    """
    return filter_subs_to_exact_sector(resolved_subs, target_sector)


def action_introduces_higher_sector(cached_eq, target):
    """DEPRECATED: OLD filtering hardcoded for sector 53.

    Only blocks integrals in 'higher sector' (top sector + extra propagators).
    This allows lateral sectors (same level, different propagators).
    Only blocks integrals that are in top sector AND have propagators at positions 1, 3, or 6.

    TODO: This function uses is_higher_sector() which is hardcoded for sector 53.
    Use action_introduces_outside_sector() with filter_mode='subsector' instead.
    """
    for integral in cached_eq.keys():
        if is_higher_sector(integral):
            return True
    return False


def action_introduces_higher_sector_general(cached_eq, target):
    """Check if action introduces integrals in sectors strictly higher than target's sector.

    A sector is higher if it has ALL the propagators the target has, PLUS extra ones.
    This allows lateral sectors (same level, different propagators) but blocks
    sectors with additional propagators on top of what target has.

    Args:
        cached_eq: IBP equation after substitutions
        target: The target integral we're reducing

    Returns:
        True if any integral is in a strictly higher sector than target
    """
    # Get target's sector mask (which propagators are >= 1)
    target_mask = tuple(1 if target[i] >= 1 else 0 for i in range(N_DENOMINATORS))

    for integral in cached_eq.keys():
        # First check if integral is "in the sector" - has all required propagators
        in_sector = True
        for i in range(N_DENOMINATORS):
            if target_mask[i] == 1 and integral[i] < 1:
                in_sector = False
                break

        if not in_sector:
            continue  # Skip integrals not in sector (lateral sectors are OK)

        # Now check if it's a HIGHER sector (has extra propagators)
        for i in range(N_DENOMINATORS):
            if target_mask[i] == 0 and integral[i] >= 1:
                return True  # Higher sector - has extra propagator
        # Also check ISPs - if integral has any ISP active, it's higher.
        if any(integral[p] >= 1 for p in ISP_POSITIONS):
            return True
    return False


def enumerate_valid_actions(target, subs, ibp_t, li_t, shifts, filter_mode='subsector'):
    """Enumerate all valid (ibp_op, delta) pairs that can eliminate target.

    Args:
        filter_mode: 'subsector' (default) - strict filtering, no lateral sectors
                     'higher_only' - blocks only strictly higher sectors, allows lateral
                     'none' - no sector filtering
    """
    # Choose filter function based on mode
    if filter_mode == 'subsector':
        should_filter = lambda eq, tgt: action_introduces_outside_sector(eq, tgt)
    elif filter_mode == 'higher_only':
        should_filter = lambda eq, tgt: action_introduces_higher_sector_general(eq, tgt)
    elif filter_mode == 'none':
        should_filter = lambda eq, tgt: False
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")
    valid = []
    seen = set()

    # Direct actions
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
                if sub_int not in raw or raw[sub_int] == 0:
                    continue
                if target in raw and raw[target] != 0:
                    continue
                cached = apply_all_substitutions(raw, subs)
                if target not in cached or cached[target] == 0:
                    continue
                if should_filter(cached, target):
                    continue
                delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid


def enumerate_valid_actions_cached(target, subs, ibp_t, li_t, shifts, filter_mode, raw_eq_cache):
    """Enumerate valid actions using a shared cache for raw equations.

    This version uses a cache dict to avoid recomputing get_raw_equation for the same
    (ibp_op, seed) pairs across multiple calls.
    """
    # Choose filter function based on mode
    if filter_mode == 'subsector':
        should_filter = lambda eq, tgt: action_introduces_outside_sector(eq, tgt)
    elif filter_mode == 'higher_only':
        should_filter = lambda eq, tgt: action_introduces_higher_sector_general(eq, tgt)
    elif filter_mode == 'none':
        should_filter = lambda eq, tgt: False
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    valid = []
    seen = set()

    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Direct actions
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = get_raw_cached(ibp_op, seed)
                if sub_int not in raw or raw[sub_int] == 0:
                    continue
                if target in raw and raw[target] != 0:
                    continue
                cached = apply_all_substitutions(raw, subs)
                if target not in cached or cached[target] == 0:
                    continue
                if should_filter(cached, target):
                    continue
                delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid


def enumerate_valid_actions_resolved(target, subs, resolved_subs, ibp_t, li_t, shifts, filter_mode, raw_eq_cache):
    """Enumerate valid actions using pre-resolved substitutions.

    OPTIMIZED VERSION: Uses apply_resolved_subs (single pass) instead of
    apply_all_substitutions (iterate to fixed point).

    Args:
        resolved_subs: Pre-computed from resolve_subs(subs). Values don't contain any keys.
    """
    # Choose filter function based on mode
    if filter_mode == 'subsector':
        should_filter = lambda eq, tgt: action_introduces_outside_sector(eq, tgt)
    elif filter_mode == 'higher_only':
        should_filter = lambda eq, tgt: action_introduces_higher_sector_general(eq, tgt)
    elif filter_mode == 'none':
        should_filter = lambda eq, tgt: False
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    valid = []
    seen = set()

    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Direct actions
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            # Use single-pass application with resolved subs
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = get_raw_cached(ibp_op, seed)
                if sub_int not in raw or raw[sub_int] == 0:
                    continue
                if target in raw and raw[target] != 0:
                    continue
                # Use single-pass application with resolved subs
                cached = apply_resolved_subs(raw, resolved_subs)
                if target not in cached or cached[target] == 0:
                    continue
                if should_filter(cached, target):
                    continue
                delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid


def apply_resolved_subs_batch(raw_eqs, resolved_subs, verbose_timing=False):
    """Apply resolved subs to multiple raw equations efficiently.

    OPTIMIZED: Deduplicates raw equations to avoid redundant computation.

    Args:
        raw_eqs: list of raw equation dicts
        resolved_subs: pre-resolved subs dict
        verbose_timing: if True, print timing breakdown

    Returns:
        list of substituted equation dicts
    """
    import time as _time
    t_start = _time.time()

    if not resolved_subs:
        # No subs to apply - just copy
        return [dict(raw) for raw in raw_eqs]

    # Deduplicate by object id - same raw eq object means same result
    t_dedup = _time.time()
    unique_cache = {}  # id(raw) -> result
    results = []

    t_dedup_elapsed = _time.time() - t_dedup
    t_apply = _time.time()
    n_computed = 0
    n_reused = 0
    total_subs_applied = 0

    for raw in raw_eqs:
        raw_id = id(raw)
        if raw_id in unique_cache:
            # Reuse previously computed result (copy it)
            results.append(dict(unique_cache[raw_id]))
            n_reused += 1
        else:
            # Compute fresh
            result = dict(raw)
            for integral in list(result.keys()):
                if integral in resolved_subs:
                    total_subs_applied += 1
                    coeff = result.pop(integral)
                    for repl_integral, repl_coeff in resolved_subs[integral].items():
                        new_coeff = (coeff * repl_coeff) % PRIME
                        if repl_integral in result:
                            result[repl_integral] = (result[repl_integral] + new_coeff) % PRIME
                        else:
                            result[repl_integral] = new_coeff
                        if result[repl_integral] == 0:
                            del result[repl_integral]
            unique_cache[raw_id] = result
            results.append(dict(result))  # Return a copy
            n_computed += 1

    t_apply_elapsed = _time.time() - t_apply
    total_elapsed = _time.time() - t_start

    if verbose_timing:
        print(f"        [apply_batch] total={total_elapsed:.4f}s | "
              f"computed={n_computed}, reused={n_reused}, subs_applied={total_subs_applied}", flush=True)

    return results


def _substitute_one_in_cached(cached, sub_key, replacement):
    """Substitute sub_key -> replacement in cached, returning a NEW dict.
    If sub_key not in cached, returns the original cached (shared reference).
    """
    if sub_key not in cached:
        return cached
    new_cached = dict(cached)
    coeff = new_cached.pop(sub_key)
    for k, v in replacement.items():
        nc = (coeff * v) % PRIME
        if k in new_cached:
            s = (new_cached[k] + nc) % PRIME
            if s == 0:
                del new_cached[k]
            else:
                new_cached[k] = s
        elif nc != 0:
            new_cached[k] = nc
    return new_cached


def compute_indirect_substituted_incremental(prev_aux, new_sub_int, new_resolved_sol,
                                              new_resolved_subs, ibp_t, li_t, shifts,
                                              raw_eq_cache):
    """Incrementally update indirect_cache state for one new substitution.

    Given prev_aux = (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
    from a previous compute (at step N with resolved_subs not yet containing
    new_sub_int), build the state at step N+1 where new_sub_int has been
    added with new_resolved_sol = (already-resolved-against-old-resolved_subs).

    Correctness: NEW resolved_subs equals OLD plus the new entry. For each
    old cached (which has no keys from OLD), applying NEW is equivalent to
    substituting just new_sub_int -> new_resolved_sol (we can prove the
    OLD-key values that changed are not in old cached).

    Returns: (result, new_aux) in same shape as compute_indirect_substituted +
             returned aux for chaining further incremental calls.
    """
    prev_cu, prev_ubm, prev_rid, prev_iraws = prev_aux

    # Phase A: update existing cached entries to substitute new_sub_int.
    new_cu = []
    new_ubm = []
    for old_c, old_ub in zip(prev_cu, prev_ubm):
        new_c = _substitute_one_in_cached(old_c, new_sub_int, new_resolved_sol)
        if new_c is old_c:
            new_cu.append(old_c)
            new_ubm.append(old_ub)
        else:
            new_cu.append(new_c)
            new_ubm.append(cached_union_bitmask(new_c))

    new_rid = dict(prev_rid)

    # Phase B: add new raws coming from new_sub_int.
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    def _get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    if new_sub_int in sub_cache:
        new_raw_list = sub_cache[new_sub_int]
    else:
        new_raw_list = []
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(new_sub_int[i] - shift[i] for i in range(N_INDICES))
                raw = _get_raw_cached(ibp_op, seed)
                if new_sub_int in raw and raw[new_sub_int] != 0:
                    new_raw_list.append((ibp_op, shift, raw))
        sub_cache[new_sub_int] = new_raw_list

    new_iraws_for_new = [(new_sub_int, op, sh, r) for op, sh, r in new_raw_list]

    raws_to_apply = []
    for sub_int, op, sh, raw in new_iraws_for_new:
        rid = id(raw)
        if rid not in new_rid:
            # Reserve the slot index BEFORE applying. Multiple unique raws
            # each get their own index — must add len(raws_to_apply) so they
            # don't all collapse to the same len(new_cu) value.
            new_rid[rid] = len(new_cu) + len(raws_to_apply)
            raws_to_apply.append(raw)

    if raws_to_apply:
        new_cacheds = apply_resolved_subs_batch(raws_to_apply, new_resolved_subs)
        for c in new_cacheds:
            new_cu.append(c)
            new_ubm.append(cached_union_bitmask(c))

    new_iraws = prev_iraws + new_iraws_for_new

    # Build result list (same format as compute_indirect_substituted).
    result = []
    for sub_int, op, sh, raw in new_iraws:
        idx = new_rid[id(raw)]
        result.append((sub_int, op, sh, raw, new_cu[idx], new_ubm[idx]))

    return result, (new_cu, new_ubm, new_rid, new_iraws)


def compute_indirect_substituted(subs, resolved_subs, ibp_t, li_t, shifts, raw_eq_cache):
    """Precompute substituted indirect raw equations for a state.

    This can be shared across all targets of the same state, avoiding
    redundant substitution computation.

    Args:
        subs: original subs dict
        resolved_subs: pre-resolved subs dict
        ibp_t, li_t: IBP and LI templates
        shifts: shifts lookup
        raw_eq_cache: cache for raw equations

    Returns:
        list of (sub_int, ibp_op, shift, raw, cached, union_bm) tuples.
        (Use compute_indirect_substituted_with_aux to also get aux state
        for incremental chaining.)

    When env BEAM_PROFILE_CIC is set, accumulates per-phase timings into
    the module-global list `_CIC_PROFILE`.
    """
    import os, time as _time
    _profile = bool(os.environ.get("BEAM_PROFILE_CIC"))
    _t0 = _time.time()

    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Initialize sub_int cache if not present
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    # Phase 1: collect indirect raw equations
    _t_phase = _time.time()
    indirect_raws = []  # List of (sub_int, ibp_op, shift, raw)
    seen_raws = set()  # Track unique raws by id to avoid duplicate subs application
    n_sub_cache_misses = 0

    for sub_int in subs:
        if sub_int in sub_cache:
            raw_list = sub_cache[sub_int]
        else:
            n_sub_cache_misses += 1
            raw_list = []
            for ibp_op, shift_list in shifts.items():
                for shift in shift_list:
                    seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                    raw = get_raw_cached(ibp_op, seed)
                    if sub_int in raw and raw[sub_int] != 0:
                        raw_list.append((ibp_op, shift, raw))
            sub_cache[sub_int] = raw_list

        for ibp_op, shift, raw in raw_list:
            indirect_raws.append((sub_int, ibp_op, shift, raw))
            seen_raws.add(id(raw))
    t_phase1 = _time.time() - _t_phase

    if not indirect_raws:
        if _profile:
            globals().setdefault('_CIC_PROFILE', []).append({
                't_total': _time.time() - _t0, 't_phase1_collect': t_phase1,
                't_phase2_dedup': 0.0, 't_phase3_apply_subs': 0.0,
                't_phase4_union_bm': 0.0, 't_phase5_build': 0.0,
                'n_indirect_raws': 0, 'n_unique_raws': 0,
                'n_sub_cache_misses': n_sub_cache_misses})
        return []

    # Phase 2: deduplicate raws by id
    _t_phase = _time.time()
    unique_raws = []
    raw_id_to_idx = {}
    for sub_int, ibp_op, shift, raw in indirect_raws:
        raw_id = id(raw)
        if raw_id not in raw_id_to_idx:
            raw_id_to_idx[raw_id] = len(unique_raws)
            unique_raws.append(raw)
    t_phase2 = _time.time() - _t_phase

    # Phase 3: apply subs to unique raws
    _t_phase = _time.time()
    cached_unique = apply_resolved_subs_batch(unique_raws, resolved_subs)
    t_phase3 = _time.time() - _t_phase

    # Phase 4: compute OR-union sector bitmasks
    _t_phase = _time.time()
    union_bms = [cached_union_bitmask(c) for c in cached_unique]
    t_phase4 = _time.time() - _t_phase

    # Phase 5: build result list
    _t_phase = _time.time()
    result = []
    for sub_int, ibp_op, shift, raw in indirect_raws:
        idx = raw_id_to_idx[id(raw)]
        cached = cached_unique[idx]
        union_bm = union_bms[idx]
        result.append((sub_int, ibp_op, shift, raw, cached, union_bm))
    t_phase5 = _time.time() - _t_phase

    if _profile:
        globals().setdefault('_CIC_PROFILE', []).append({
            't_total': _time.time() - _t0,
            't_phase1_collect': t_phase1,
            't_phase2_dedup': t_phase2,
            't_phase3_apply_subs': t_phase3,
            't_phase4_union_bm': t_phase4,
            't_phase5_build': t_phase5,
            'n_indirect_raws': len(indirect_raws),
            'n_unique_raws': len(unique_raws),
            'n_sub_cache_misses': n_sub_cache_misses,
        })

    return result


def compute_indirect_substituted_with_aux(subs, resolved_subs, ibp_t, li_t,
                                            shifts, raw_eq_cache):
    """Full compute that ALSO returns aux state for chaining incremental updates.

    Returns: (result, aux_state) where
        aux_state = (cached_unique, union_bms, raw_id_to_idx, indirect_raws)

    Pass aux_state to compute_indirect_substituted_incremental on the next step.
    """
    # Reuse the same logic by recomputing locally — we need access to the
    # intermediate variables that the standard function doesn't expose. The
    # cost of this is identical to compute_indirect_substituted itself.
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    def _grc(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    indirect_raws = []
    for sub_int in subs:
        if sub_int in sub_cache:
            raw_list = sub_cache[sub_int]
        else:
            raw_list = []
            for ibp_op, shift_list in shifts.items():
                for shift in shift_list:
                    seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                    raw = _grc(ibp_op, seed)
                    if sub_int in raw and raw[sub_int] != 0:
                        raw_list.append((ibp_op, shift, raw))
            sub_cache[sub_int] = raw_list
        for ibp_op, shift, raw in raw_list:
            indirect_raws.append((sub_int, ibp_op, shift, raw))

    if not indirect_raws:
        return [], ([], [], {}, [])

    unique_raws = []
    raw_id_to_idx = {}
    for sub_int, ibp_op, shift, raw in indirect_raws:
        rid = id(raw)
        if rid not in raw_id_to_idx:
            raw_id_to_idx[rid] = len(unique_raws)
            unique_raws.append(raw)

    cached_unique = apply_resolved_subs_batch(unique_raws, resolved_subs)
    union_bms = [cached_union_bitmask(c) for c in cached_unique]

    result = []
    for sub_int, ibp_op, shift, raw in indirect_raws:
        idx = raw_id_to_idx[id(raw)]
        result.append((sub_int, ibp_op, shift, raw, cached_unique[idx], union_bms[idx]))

    return result, (cached_unique, union_bms, raw_id_to_idx, indirect_raws)


def enumerate_valid_actions_with_indirect_cache(target, indirect_cache, subs, resolved_subs, ibp_t, li_t, shifts, filter_mode, raw_eq_cache, verbose_timing=False):
    """Enumerate valid actions using precomputed indirect cache.

    OPTIMIZED: Uses precomputed substituted indirect raw equations shared
    across all targets of the same state.

    Args:
        target: target integral to eliminate
        indirect_cache: precomputed list from compute_indirect_substituted
        subs, resolved_subs, ibp_t, li_t, shifts, filter_mode, raw_eq_cache: as before
        verbose_timing: if True, print timing info

    Returns:
        list of (ibp_op, delta) tuples representing valid actions

    When env BEAM_PROFILE_CSV is set, accumulates fine-grained per-call
    counters (#entries, #passing each short-circuit, time per phase,
    avg cached-dict size for surviving entries) into a module-global
    dict `_GETVALID_PROFILE` keyed by call_id. The caller can flush
    these to a sibling profile CSV.
    """
    import time as _time
    import os
    _detailed_profile = bool(os.environ.get("BEAM_PROFILE_CSV"))
    t_start = _time.time()

    # Choose filter function
    if filter_mode == 'subsector':
        should_filter = lambda eq, tgt: action_introduces_outside_sector(eq, tgt)
    elif filter_mode == 'higher_only':
        should_filter = lambda eq, tgt: action_introduces_higher_sector_general(eq, tgt)
    elif filter_mode == 'none':
        should_filter = lambda eq, tgt: False
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    # Fast path: for the 'subsector' filter, the check is equivalent to a
    # single bitwise op (union_bm & ~target_bm != 0). The union_bm is
    # precomputed once per indirect_cache entry (see compute_indirect_substituted).
    fast_subsector_filter = (filter_mode == 'subsector')
    target_bm = sector_bitmask(target) if fast_subsector_filter else None
    not_target_bm = (~target_bm) if fast_subsector_filter else None

    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Phase 1a: Direct actions (target-dependent)
    t1a = _time.time()
    valid = []
    seen = set()

    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            # Apply subs and check
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            if fast_subsector_filter:
                if (cached_union_bitmask(cached) & not_target_bm) != 0:
                    continue
            else:
                if should_filter(cached, target):
                    continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))
    t1a_elapsed = _time.time() - t1a

    # Phase 1b: Filter precomputed indirect cache (target-dependent filtering only)
    t1b = _time.time()
    n_indirect_checked = 0
    n_indirect_valid = 0
    n_pass_sc1 = 0           # passed "target not in raw" short-circuit (we keep going)
    n_pass_sc2 = 0           # passed "target in cached" short-circuit
    n_pass_filter = 0        # passed sector filter
    t_sc1 = 0.0              # time on the target-in-raw check
    t_sc2 = 0.0              # time on the target-in-cached check
    t_filter_in_loop = 0.0   # time inside should_filter calls
    sum_cached_size = 0      # cumulative |cached| for entries that hit sc2
    sum_should_filter_cached_size = 0  # cumulative |cached| for entries that reached filter

    for sub_int, ibp_op, shift, raw, cached, union_bm in indirect_cache:
        n_indirect_checked += 1
        if _detailed_profile:
            _t = _time.time()
            in_raw = target in raw and raw[target] != 0
            t_sc1 += _time.time() - _t
            if in_raw:
                continue
        else:
            if target in raw and raw[target] != 0:
                continue
        n_pass_sc1 += 1
        if _detailed_profile:
            _t = _time.time()
            sc2_fail = target not in cached or cached[target] == 0
            t_sc2 += _time.time() - _t
            if sc2_fail:
                continue
        else:
            if target not in cached or cached[target] == 0:
                continue
        n_pass_sc2 += 1
        if _detailed_profile:
            sum_cached_size += len(cached)
            sum_should_filter_cached_size += len(cached)
            _t = _time.time()
            if fast_subsector_filter:
                filt = (union_bm & not_target_bm) != 0
            else:
                filt = should_filter(cached, target)
            t_filter_in_loop += _time.time() - _t
            if filt:
                continue
        else:
            if fast_subsector_filter:
                if (union_bm & not_target_bm) != 0:
                    continue
            else:
                if should_filter(cached, target):
                    continue
        n_pass_filter += 1
        seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
        delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
        if (ibp_op, delta) not in seen:
            seen.add((ibp_op, delta))
            valid.append((ibp_op, delta))
            n_indirect_valid += 1
    t1b_elapsed = _time.time() - t1b

    total_elapsed = _time.time() - t_start

    if _detailed_profile:
        prof = globals().setdefault('_GETVALID_PROFILE', [])
        prof.append({
            'n_indirect_checked': n_indirect_checked,
            'n_pass_sc1': n_pass_sc1,
            'n_pass_sc2': n_pass_sc2,
            'n_pass_filter': n_pass_filter,
            't_sc1': t_sc1,
            't_sc2': t_sc2,
            't_filter_in_loop': t_filter_in_loop,
            't_p1a_direct': t1a_elapsed,
            't_p1b_indirect': t1b_elapsed,
            't_total': total_elapsed,
            'sum_cached_size_at_sc2': sum_cached_size,
            'sum_cached_size_at_filter': sum_should_filter_cached_size,
        })

    if verbose_timing:
        print(f"      [enumerate_cached] total={total_elapsed:.3f}s | "
              f"P1a(direct)={t1a_elapsed:.3f}s | "
              f"P1b(indirect_filter)={t1b_elapsed:.3f}s ({n_indirect_valid}/{n_indirect_checked}) | "
              f"valid={len(valid)}", flush=True)

    return valid


def enumerate_valid_actions_batched(target, subs, resolved_subs, ibp_t, li_t, shifts, filter_mode, raw_eq_cache, verbose_timing=False):
    """Batched version of enumerate_valid_actions_resolved.

    OPTIMIZED: Collects all candidate raw equations first, then applies subs
    in batch to reduce function call overhead. ~3-4x faster than non-batched.

    Args:
        target: target integral to eliminate
        subs: original subs dict
        resolved_subs: pre-resolved subs dict
        ibp_t, li_t: IBP and LI templates
        shifts: shifts lookup
        filter_mode: sector filtering mode
        raw_eq_cache: cache for raw equations
        verbose_timing: if True, print detailed timing info

    Returns:
        list of (ibp_op, delta) tuples representing valid actions
    """
    import time as _time
    t_start = _time.time()

    # Choose filter function based on mode
    if filter_mode == 'subsector':
        should_filter = lambda eq, tgt: action_introduces_outside_sector(eq, tgt)
    elif filter_mode == 'higher_only':
        should_filter = lambda eq, tgt: action_introduces_higher_sector_general(eq, tgt)
    elif filter_mode == 'none':
        should_filter = lambda eq, tgt: False
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Phase 1a: Direct actions
    t1a = _time.time()
    candidates = []  # List of (ibp_op, delta, raw_eq)
    n_direct_checked = 0
    n_direct_found = 0

    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            n_direct_checked += 1
            seed = tuple(target[i] - shift[i] for i in range(N_INDICES))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            candidates.append((ibp_op, delta, raw))
            n_direct_found += 1
    t1a_elapsed = _time.time() - t1a

    # Phase 1b: Indirect actions
    # OPTIMIZATION: Cache which (ibp_op, shift, raw) contain each sub_int
    # This avoids re-checking sub_int membership in raw for each target
    t1b = _time.time()
    n_subs = len(subs)
    n_indirect_checked = 0
    n_indirect_found = 0
    n_cache_hits = 0
    n_sub_cache_hits = 0
    cache_size_before = len(raw_eq_cache)

    # Initialize sub_int cache if not present (stored in raw_eq_cache to persist)
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    for sub_int in subs:
        # Check if we've already computed which raw eqs contain this sub_int
        if sub_int in sub_cache:
            n_sub_cache_hits += 1
            raw_list = sub_cache[sub_int]
        else:
            # Compute and cache: find all (ibp_op, shift, raw) where sub_int is in raw
            raw_list = []
            for ibp_op, shift_list in shifts.items():
                for shift in shift_list:
                    seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
                    raw = get_raw_cached(ibp_op, seed)
                    if sub_int in raw and raw[sub_int] != 0:
                        raw_list.append((ibp_op, shift, raw))
            sub_cache[sub_int] = raw_list

        # Now check target against the precomputed raw equations
        for ibp_op, shift, raw in raw_list:
            n_indirect_checked += 1
            if target in raw and raw[target] != 0:
                continue
            seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
            delta = tuple(seed[i] - target[i] for i in range(N_INDICES))
            candidates.append((ibp_op, delta, raw))
            n_indirect_found += 1
    t1b_elapsed = _time.time() - t1b
    cache_size_after = len(raw_eq_cache) - 1  # Subtract 1 for _sub_cache key

    if not candidates:
        if verbose_timing:
            print(f"      [enumerate] no candidates, subs={n_subs}, direct={n_direct_found}/{n_direct_checked}, indirect={n_indirect_found}/{n_indirect_checked}")
        return []

    # Phase 2: Apply subs to all candidates in batch
    t2 = _time.time()
    raw_eqs = [c[2] for c in candidates]
    n_unique_raw = len(set(id(r) for r in raw_eqs))

    # Count total terms in resolved_subs
    n_resolved_subs = len(resolved_subs)
    n_resolved_terms = sum(len(v) for v in resolved_subs.values())

    cached_eqs = apply_resolved_subs_batch(raw_eqs, resolved_subs, verbose_timing=verbose_timing)
    t2_elapsed = _time.time() - t2

    # Phase 3: Filter results and collect valid actions
    t3 = _time.time()
    valid = []
    seen = set()
    n_filtered_no_target = 0
    n_filtered_sector = 0
    for i, (ibp_op, delta, raw) in enumerate(candidates):
        cached = cached_eqs[i]
        if target not in cached or cached[target] == 0:
            n_filtered_no_target += 1
            continue
        if should_filter(cached, target):
            n_filtered_sector += 1
            continue
        if (ibp_op, delta) not in seen:
            seen.add((ibp_op, delta))
            valid.append((ibp_op, delta))
    t3_elapsed = _time.time() - t3

    total_elapsed = _time.time() - t_start

    if verbose_timing:
        print(f"      [enumerate] total={total_elapsed:.3f}s | "
              f"P1a(direct)={t1a_elapsed:.3f}s ({n_direct_found}/{n_direct_checked}) | "
              f"P1b(indirect)={t1b_elapsed:.3f}s ({n_indirect_found}/{n_indirect_checked}, subs={n_subs}, sub_cache_hits={n_sub_cache_hits}) | "
              f"P2(apply_subs)={t2_elapsed:.3f}s (cands={len(candidates)}, unique_raw={n_unique_raw}, resolved={n_resolved_subs}/{n_resolved_terms}terms) | "
              f"P3(filter)={t3_elapsed:.3f}s (no_tgt={n_filtered_no_target}, sector={n_filtered_sector}, valid={len(valid)}) | "
              f"raw_cache_size={cache_size_after}", flush=True)

    return valid


class IBPEnvironment:
    """Environment for IBP reduction with model-based action selection."""

    def __init__(self, ibp_path=None, li_path=None):
        # If explicit paths are passed (legacy callers), parse them. Otherwise
        # use the already-loaded templates from the current topology (set by
        # init_from_topology). This is the topology-agnostic path.
        if ibp_path is not None:
            self.ibp_t = parse_templates(ibp_path)
        elif IBP_TEMPLATES:
            self.ibp_t = dict(IBP_TEMPLATES)
        else:
            # Fall back to the legacy default path (trianglebox).
            self.ibp_t = parse_templates(IBP_PATH)
        if li_path is not None:
            self.li_t = parse_templates(li_path)
        elif LI_TEMPLATES:
            self.li_t = dict(LI_TEMPLATES)
        else:
            self.li_t = parse_templates(LI_PATH)

        # Build shifts lookup. Action indices 0..n_ibp-1 are IBPs,
        # n_ibp..n_ibp+n_li-1 are LIs. Derived from the loaded templates
        # so this works for any topology (trianglebox: 8+1, pentagon: 12+6).
        n_ibp = len(self.ibp_t)
        self.shifts = {}
        for ibp_op in range(n_ibp):
            if ibp_op in self.ibp_t:
                self.shifts[ibp_op] = [s for s, _ in self.ibp_t[ibp_op]]
        for li_idx in self.li_t:
            self.shifts[n_ibp + li_idx] = [s for s, _ in self.li_t[li_idx]]

        # Cache for get_raw_equation results: (ibp_op, seed) -> equation dict
        self._raw_eq_cache = {}

    def get_raw_equation_cached(self, ibp_op, seed):
        """Cached version of get_raw_equation."""
        key = (ibp_op, seed)
        if key not in self._raw_eq_cache:
            self._raw_eq_cache[key] = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
        return self._raw_eq_cache[key]

    def get_valid_actions_cached(self, target, subs, filter_mode='subsector'):
        """Get list of valid actions for target, using cached raw equations."""
        return enumerate_valid_actions_cached(
            target, subs, self.ibp_t, self.li_t, self.shifts, filter_mode,
            self._raw_eq_cache
        )

    def get_valid_actions_resolved(self, target, subs, resolved_subs, filter_mode='subsector'):
        """Get valid actions using pre-resolved substitutions.

        OPTIMIZED: Uses single-pass substitution instead of iterating to fixed point.

        Args:
            resolved_subs: Pre-computed from resolve_subs(subs).
        """
        return enumerate_valid_actions_resolved(
            target, subs, resolved_subs, self.ibp_t, self.li_t, self.shifts, filter_mode,
            self._raw_eq_cache
        )

    def get_valid_actions_batched(self, target, subs, resolved_subs, filter_mode='subsector', verbose_timing=False):
        """Get valid actions using batched substitution application.

        OPTIMIZED: ~3-4x faster than get_valid_actions_resolved by batching
        all substitution applications together.

        Args:
            resolved_subs: Pre-computed from resolve_subs(subs).
            verbose_timing: if True, print detailed timing info for profiling.
        """
        return enumerate_valid_actions_batched(
            target, subs, resolved_subs, self.ibp_t, self.li_t, self.shifts, filter_mode,
            self._raw_eq_cache, verbose_timing=verbose_timing
        )

    def compute_indirect_cache(self, subs, resolved_subs):
        """Precompute substituted indirect raw equations for a state.

        Call this once per state, then use get_valid_actions_with_cache for each target.
        This avoids redundant substitution computation across targets of the same state.

        Returns:
            indirect_cache: opaque object to pass to get_valid_actions_with_cache
        """
        return compute_indirect_substituted(
            subs, resolved_subs, self.ibp_t, self.li_t, self.shifts, self._raw_eq_cache
        )

    def get_valid_actions_with_cache(self, target, indirect_cache, subs, resolved_subs, filter_mode='subsector', verbose_timing=False):
        """Get valid actions using precomputed indirect cache.

        OPTIMIZED: Uses precomputed substituted indirect raw equations.
        Call compute_indirect_cache once per state, then this for each target.

        Args:
            indirect_cache: from compute_indirect_cache
            Other args: as in get_valid_actions_batched
        """
        return enumerate_valid_actions_with_indirect_cache(
            target, indirect_cache, subs, resolved_subs, self.ibp_t, self.li_t, self.shifts,
            filter_mode, self._raw_eq_cache, verbose_timing=verbose_timing
        )

    def apply_action(self, expr, subs, target, ibp_op, delta):
        """
        Apply action to eliminate target.

        Returns:
            (new_expr, new_subs, success)
        """
        seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
        raw = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
        cached = apply_all_substitutions(raw, subs)

        if target not in cached or cached[target] == 0:
            return expr, subs, False

        sol = solve_ibp_for(cached, target)
        if sol is None:
            return expr, subs, False

        new_subs = dict(subs)
        new_subs[target] = sol
        new_expr = apply_substitution(expr, target, sol)

        return new_expr, new_subs, True

    def apply_action_resolved(self, expr, subs, resolved_subs, target, ibp_op, delta):
        """Apply action using pre-resolved substitutions.

        OPTIMIZED: Uses single-pass substitution.

        Returns:
            (new_expr, new_subs, new_resolved_subs, success)
        """
        seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
        raw = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
        # Use single-pass application
        cached = apply_resolved_subs(raw, resolved_subs)

        if target not in cached or cached[target] == 0:
            return expr, subs, resolved_subs, False

        sol = solve_ibp_for(cached, target)
        if sol is None:
            return expr, subs, resolved_subs, False

        new_subs = dict(subs)
        new_subs[target] = sol
        new_expr = apply_substitution(expr, target, sol)

        # Incrementally update resolved subs (O(|subs|) instead of O(|subs|²))
        new_resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)

        return new_expr, new_subs, new_resolved_subs, True

    def apply_action_resolved_target_only(self, expr_t, subs, resolved_subs,
                                            target, ibp_op, delta, target_sector):
        """Like apply_action_resolved but tracks ONLY target-sector content
        in expr AND in subs/resolved_subs.

        Sub-sector contributions are stripped from sol before storing in subs
        and resolved_subs. This is safe for beam evolution because:
          - target-sector(new_expr_t) depends only on target-sector(cached),
            which depends only on target-sector portions of resolved_subs values
          - the model context doesn't use subs values to affect predictions
          - therefore stripping sub-sector content from resolved_subs values
            doesn't change the next step's target-sector beam evolution

        The caller is responsible for reconstructing the FULL final_expr
        (including sub-sector content) at worker end by replaying the path
        through raw IBP templates from scratch (see replay_path_to_full_expr).

        Returns:
            (new_expr_t, new_subs, new_resolved_subs, success)
        """
        seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
        raw = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
        cached = apply_resolved_subs(raw, resolved_subs)

        if target not in cached or cached[target] == 0:
            return expr_t, subs, resolved_subs, False

        sol = solve_ibp_for(cached, target)
        if sol is None:
            return expr_t, subs, resolved_subs, False

        # Strip sol to target-sector before storing. target-sector(new_expr_t)
        # is unaffected because apply_substitution_target_only already drops
        # non-target terms. Storing the stripped version keeps subs and
        # resolved_subs strictly target-sector-content-only.
        sol_target = {k: v for k, v in sol.items()
                      if integral_in_exact_sector(k, target_sector)}

        new_subs = dict(subs)
        new_subs[target] = sol_target
        new_expr_t = apply_substitution_target_only(expr_t, target, sol_target, target_sector)
        new_resolved_subs = add_sub_to_resolved(resolved_subs, target, sol_target)

        return new_expr_t, new_subs, new_resolved_subs, True

    def replay_path_to_full_expr(self, start_expr, path):
        """Reconstruct the FULL final_expr (target + sub-sector) by replaying
        path through raw IBP templates from scratch.

        Under Option F + resolved_subs stripping, beam_search only tracks
        target-sector content. To recover the complete final expression for
        the orchestrator's cache, the WINNING trajectory's path is replayed
        here against fresh raw IBPs, rebuilding full subs/resolved_subs only
        for this one path (cheap, since it's a single trajectory, not the
        whole beam).

        Args:
            start_expr: initial expression passed into beam_search (typically
                a single-integral dict like {input_integral: 1}).
            path: list of (target, ibp_op, delta) tuples produced by beam_search.

        Returns:
            full_expr dict (integral -> coeff mod PRIME), with both
            target-sector AND sub-sector content correctly accumulated.
        """
        full_resolved_subs = {}
        full_expr = dict(start_expr)
        for target, ibp_op, delta in path:
            seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
            raw = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
            cached = apply_resolved_subs(raw, full_resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            sol = solve_ibp_for(cached, target)
            if sol is None:
                continue
            full_resolved_subs = add_sub_to_resolved(full_resolved_subs, target, sol)
            full_expr = apply_substitution(full_expr, target, sol)
        return full_expr

    def apply_action_resolved_split(self, expr_t, sub_accum, subs, resolved_subs,
                                     target, ibp_op, delta, target_sector):
        """Same as apply_action_resolved but with sub-sector content factored
        out of expr. expr_t is target-sector content only; sub_accum holds the
        passenger sub-sector terms.

        Saves O(|sub_accum|) per call vs apply_action_resolved at late stages.

        Returns:
            (new_expr_t, new_sub_accum, new_subs, new_resolved_subs, success)
        """
        seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
        raw = get_raw_equation(self.ibp_t, self.li_t, ibp_op, seed)
        cached = apply_resolved_subs(raw, resolved_subs)

        if target not in cached or cached[target] == 0:
            return expr_t, sub_accum, subs, resolved_subs, False

        sol = solve_ibp_for(cached, target)
        if sol is None:
            return expr_t, sub_accum, subs, resolved_subs, False

        new_subs = dict(subs)
        new_subs[target] = sol
        new_expr_t, new_sub_accum = apply_substitution_split(
            expr_t, sub_accum, target, sol, target_sector
        )

        new_resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)

        return new_expr_t, new_sub_accum, new_subs, new_resolved_subs, True

    def get_valid_actions(self, target, subs, filter_mode='subsector'):
        """Get list of valid actions for target.

        Args:
            filter_mode: 'subsector' (default) - strict filtering, no lateral sectors
                         'higher_only' - old filtering, allows lateral sectors
                         'none' - no sector filtering
        """
        return enumerate_valid_actions(target, subs, self.ibp_t, self.li_t, self.shifts, filter_mode)

    def is_reduced(self, expr):
        """Check if expression is fully reduced to masters."""
        top_only = filter_top_sector(expr)
        non_masters = {k: v for k, v in top_only.items() if not is_master(k)}
        return len(non_masters) == 0

    def reduce_with_model(self, model, expr_init, device='cpu', max_steps=100, verbose=False):
        """
        Reduce expression using model to predict actions.

        Args:
            model: IBPActionClassifier model
            expr_init: Initial expression dict
            device: torch device
            max_steps: Maximum reduction steps
            verbose: Print progress

        Returns:
            (success, n_steps, final_expr)
        """
        model.eval()
        expr = dict(expr_init)
        subs = {}

        for step in range(max_steps):
            target = get_target(expr)
            if target is None:
                # Done - reduced to masters
                return True, step, expr

            # Get valid actions
            valid_actions = self.get_valid_actions(target, subs)
            if not valid_actions:
                if verbose:
                    print(f"  Step {step}: No valid actions for target {list(target)}")
                return False, step, expr

            # Prepare input for model
            batch = self._prepare_model_input(expr, subs, valid_actions, device)

            # Get model prediction
            with torch.no_grad():
                logits, _ = model(
                    batch['expr_integrals'],
                    batch['expr_coeffs'],
                    batch['expr_mask'],
                    batch['sub_integrals'],
                    batch['sub_mask'],
                    batch['action_ibp_ops'],
                    batch['action_deltas'],
                    batch['action_mask']
                )

            # Get predicted action
            pred_idx = logits.argmax(dim=-1).item()
            if pred_idx >= len(valid_actions):
                pred_idx = 0  # Fallback

            ibp_op, delta = valid_actions[pred_idx]

            # Apply action
            expr, subs, success = self.apply_action(expr, subs, target, ibp_op, tuple(delta))

            if not success:
                if verbose:
                    print(f"  Step {step}: Action failed: ibp_op={ibp_op}, delta={delta}")
                return False, step, expr

            if verbose:
                nm_count = len([k for k in filter_top_sector(expr) if not is_master(k)])
                w = weight(target)
                print(f"    Step {step}: target={list(target)} w=({w[0]},{w[1]}) | "
                      f"{len(valid_actions)} actions | chose ibp_op={ibp_op} delta={list(delta)} | "
                      f"{nm_count} non-masters left", flush=True)

        # Max steps reached
        return False, max_steps, expr

    def _prepare_model_input(self, expr, subs, valid_actions, device):
        """Prepare model input tensors for a single state."""
        max_terms = 200
        max_subs = 50
        max_actions = 900

        # Expression
        expr_integrals = torch.zeros(1, max_terms, 7, dtype=torch.long)
        expr_coeffs = torch.zeros(1, max_terms, dtype=torch.long)
        expr_mask = torch.zeros(1, max_terms, dtype=torch.bool)

        top_expr = filter_top_sector(expr)
        for j, (integral, coeff) in enumerate(list(top_expr.items())[:max_terms]):
            expr_integrals[0, j] = torch.tensor(integral)
            expr_coeffs[0, j] = coeff
            expr_mask[0, j] = True

        # Substitutions
        sub_integrals = torch.zeros(1, max_subs, 7, dtype=torch.long)
        sub_mask = torch.zeros(1, max_subs, dtype=torch.bool)

        for j, integral in enumerate(list(subs.keys())[:max_subs]):
            sub_integrals[0, j] = torch.tensor(integral)
            sub_mask[0, j] = True

        # Actions
        action_ibp_ops = torch.zeros(1, max_actions, dtype=torch.long)
        action_deltas = torch.zeros(1, max_actions, 7, dtype=torch.long)
        action_mask = torch.zeros(1, max_actions, dtype=torch.bool)

        for j, (ibp_op, delta) in enumerate(valid_actions[:max_actions]):
            action_ibp_ops[0, j] = ibp_op
            action_deltas[0, j] = torch.tensor(delta)
            action_mask[0, j] = True

        return {
            'expr_integrals': expr_integrals.to(device),
            'expr_coeffs': expr_coeffs.to(device),
            'expr_mask': expr_mask.to(device),
            'sub_integrals': sub_integrals.to(device),
            'sub_mask': sub_mask.to(device),
            'action_ibp_ops': action_ibp_ops.to(device),
            'action_deltas': action_deltas.to(device),
            'action_mask': action_mask.to(device)
        }


def evaluate_on_scrambles(model, scramble_data, env, device='cpu', n_scrambles=20, verbose=False):
    """
    Evaluate model on full reduction tasks.

    Args:
        model: IBPActionClassifier
        scramble_data: List of classifier samples (will group by scramble_id)
        env: IBPEnvironment
        device: torch device
        n_scrambles: Number of scrambles to evaluate
        verbose: Print progress

    Returns:
        dict with success_rate, avg_steps, etc.
    """
    # Group samples by scramble_id, get initial expressions
    scrambles = {}
    for sample in scramble_data:
        sid = sample['scramble_id']
        if sid not in scrambles:
            scrambles[sid] = []
        scrambles[sid].append(sample)

    # Sort each scramble by step, take step 0 for initial expression
    initial_exprs = {}
    for sid, samples in scrambles.items():
        samples.sort(key=lambda x: x['step'])
        initial_exprs[sid] = {tuple(k): v for k, v in samples[0]['expr']}

    # Evaluate on subset
    scramble_ids = list(initial_exprs.keys())[:n_scrambles]

    successes = 0
    total_steps = 0
    results = []

    for sid in scramble_ids:
        expr = initial_exprs[sid]
        success, n_steps, final_expr = env.reduce_with_model(
            model, expr, device=device, verbose=verbose
        )

        if success:
            successes += 1
        total_steps += n_steps
        results.append({
            'scramble_id': sid,
            'success': success,
            'steps': n_steps
        })

        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"Scramble {sid}: {status} in {n_steps} steps")

    return {
        'success_rate': successes / len(scramble_ids),
        'successes': successes,
        'total': len(scramble_ids),
        'avg_steps': total_steps / len(scramble_ids),
        'results': results
    }


if __name__ == '__main__':
    # Quick test
    print("Testing IBPEnvironment...")
    env = IBPEnvironment()
    print(f"Loaded {len(env.ibp_t)} IBP templates, {len(env.li_t)} LI templates")
    print(f"Shifts: {len(env.shifts)} operators")

    # Test with a simple expression
    import json
    with open('/home/shih/work/IBPreduction/data/classifier_training_data.jsonl') as f:
        sample = json.loads(f.readline())

    expr = {tuple(k): v for k, v in sample['expr']}
    print(f"\nTest expression: {len(expr)} terms")

    target = get_target(expr)
    print(f"Target: {list(target)}")

    valid = env.get_valid_actions(target, {})
    print(f"Valid actions: {len(valid)}")
