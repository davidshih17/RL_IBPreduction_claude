"""
IBP Reduction Environment for end-to-end evaluation.

Provides functions to:
1. Load IBP/LI templates
2. Apply actions to expressions
3. Enumerate valid actions
4. Run full reductions using a model

Supports configurable prime for modular arithmetic.
Use set_prime(p) to change from default (2147483647).
Smaller primes like 1009 make coefficients more learnable.
"""

import re
import torch
from pathlib import Path

# Default prime - can be changed with set_prime()
PRIME = 2147483647

# Default kinematics - can be changed with set_kinematics()
KINEMATICS = {'d': 41, 'm1': 1, 'm2': 31, 'm3': 47}

def set_prime(p):
    """Set the prime for modular arithmetic."""
    global PRIME
    PRIME = p
    print(f"IBP environment using PRIME = {PRIME}")

def set_kinematics(d=41, m1=1, m2=31, m3=47):
    """Set the kinematics (dimension and masses)."""
    global KINEMATICS
    KINEMATICS = {'d': d, 'm1': m1, 'm2': m2, 'm3': m3}
    print(f"IBP environment using KINEMATICS = d={d}, m1={m1}, m2={m2}, m3={m3}")

# Default paths
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


def parse_templates(path):
    """Parse IBP or LI template file."""
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
            if '→' in line:
                line = line.split('→', 1)[1]
            m = re.match(r'trianglebox\[([^\]]+)\]\*\(([^)]+)\)', line)
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
    """Evaluate coefficient expression with given seed."""
    a0, a1, a2, a3, a4, a5, a6 = seed
    d = KINEMATICS['d']
    m1, m2, m3 = KINEMATICS['m1'], KINEMATICS['m2'], KINEMATICS['m3']
    try:
        return eval(coeff_str.replace('^', '**')) % PRIME
    except:
        return 0


def get_raw_equation(ibp_t, li_t, ibp_op, seed):
    """Get raw IBP equation for given operator and seed."""
    if ibp_op >= 8:
        template = li_t.get(ibp_op - 8, [])
    else:
        template = ibp_t.get(ibp_op, [])
    eq = {}
    for shift, coeff_str in template:
        c = eval_coeff(coeff_str, seed)
        if c != 0:
            integral = tuple(seed[i] + shift[i] for i in range(7))
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


def is_top_sector(i):
    return i[0] >= 1 and i[2] >= 1 and i[4] >= 1 and i[5] >= 1


# The 16 master integrals from paper 2502.05121v1.pdf (equation 2.5)
PAPER_MASTERS = frozenset([
    (0, 1, 1, 1, 0, 0, 0),
    (1, 0, 1, 0, 1, 0, 0),
    (1, 1, 0, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (1, 0, 1, 1, 1, 0, 0),
    (1, -1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 0, 0),
    (-1, 1, 1, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 0, 0),
    (1, 0, 1, 0, 0, 1, 0),
    (1, 1, 0, 1, 0, 1, 0),
    (1, 0, 1, 0, 1, 1, 0),
    (1, -1, 1, 0, 1, 1, 0),
    (1, 0, 0, 1, 1, 1, 0),
    (1, 1, 0, 1, 1, 1, 0),
    (1, 0, 1, 1, 1, 1, 0),
])


def get_sector(integral):
    """Get the sector mask for an integral (first 6 indices)."""
    return tuple(1 if integral[j] > 0 else 0 for j in range(6))


# Sectors covered by the paper masters
PAPER_MASTER_SECTORS = frozenset(get_sector(m) for m in PAPER_MASTERS)

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
    - All positive indices in positions 0-5 equal to exactly 1
    - No numerator power (position 6 = 0)
    """
    for j in range(6):
        if integral[j] > 1:
            return False
        if integral[j] < 0:
            return False
    if integral[6] != 0:
        return False
    return True


def is_master(i):
    """Check if integral is a master integral.

    Masters are:
    1. The 16 specific integrals from paper 2502.05121v1.pdf (eq. 2.5)
    2. Corner integrals in sectors NOT covered by the 16 paper masters
       (unless PAPER_MASTERS_ONLY is True)
    """
    # Check if it's one of the 16 paper masters
    if i in PAPER_MASTERS:
        return True

    # If paper-masters-only mode, don't accept corner integrals
    if PAPER_MASTERS_ONLY:
        return False

    # Check if it's a corner integral in an uncovered sector
    sector = get_sector(i)
    if sector not in PAPER_MASTER_SECTORS:
        return is_corner_integral(i)

    return False


def is_higher_sector(i):
    # TODO: This is hardcoded for sector 53 (top sector = positions 0,2,4,5).
    # It checks if integral is in top sector AND has extra propagators at 1, 3, or 6.
    # If used for other sectors, this logic would need to be generalized.
    # Currently only used by filter_mode='higher_only' which is deprecated.
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
    for i in range(7):  # Check all 7 indices including ISP
        if target[i] <= 0 and integral[i] > 0:
            return False
    return True


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
    for i in range(6):
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
    for i in range(6):
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
    target_mask = tuple(1 if target[i] >= 1 else 0 for i in range(6))

    for integral in cached_eq.keys():
        # First check if integral is "in the sector" - has all required propagators
        in_sector = True
        for i in range(6):
            if target_mask[i] == 1 and integral[i] < 1:
                in_sector = False
                break

        if not in_sector:
            continue  # Skip integrals not in sector (lateral sectors are OK)

        # Now check if it's a HIGHER sector (has extra propagators)
        for i in range(6):
            if target_mask[i] == 0 and integral[i] >= 1:
                return True  # Higher sector - has extra propagator
        # Also check ISP (index 6) - if integral has ISP, it's higher
        if integral[6] >= 1:
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
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(7))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(7))
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
                delta = tuple(seed[i] - target[i] for i in range(7))
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
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(7))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(7))
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
                delta = tuple(seed[i] - target[i] for i in range(7))
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
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            # Use single-pass application with resolved subs
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(7))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions
    for sub_int in subs:
        for ibp_op, shift_list in shifts.items():
            for shift in shift_list:
                seed = tuple(sub_int[i] - shift[i] for i in range(7))
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
                delta = tuple(seed[i] - target[i] for i in range(7))
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
        list of (sub_int, ibp_op, shift, raw, cached) tuples
    """
    def get_raw_cached(ibp_op, seed):
        key = (ibp_op, seed)
        if key not in raw_eq_cache:
            raw_eq_cache[key] = get_raw_equation(ibp_t, li_t, ibp_op, seed)
        return raw_eq_cache[key]

    # Initialize sub_int cache if not present
    if '_sub_cache' not in raw_eq_cache:
        raw_eq_cache['_sub_cache'] = {}
    sub_cache = raw_eq_cache['_sub_cache']

    # Collect all unique indirect raw equations
    indirect_raws = []  # List of (sub_int, ibp_op, shift, raw)
    seen_raws = set()  # Track unique raws by id to avoid duplicate subs application

    for sub_int in subs:
        if sub_int in sub_cache:
            raw_list = sub_cache[sub_int]
        else:
            raw_list = []
            for ibp_op, shift_list in shifts.items():
                for shift in shift_list:
                    seed = tuple(sub_int[i] - shift[i] for i in range(7))
                    raw = get_raw_cached(ibp_op, seed)
                    if sub_int in raw and raw[sub_int] != 0:
                        raw_list.append((ibp_op, shift, raw))
            sub_cache[sub_int] = raw_list

        for ibp_op, shift, raw in raw_list:
            indirect_raws.append((sub_int, ibp_op, shift, raw))
            seen_raws.add(id(raw))

    if not indirect_raws:
        return []

    # Apply subs to all unique raw equations
    # Build mapping from raw id to index for deduplication
    unique_raws = []
    raw_id_to_idx = {}
    for sub_int, ibp_op, shift, raw in indirect_raws:
        raw_id = id(raw)
        if raw_id not in raw_id_to_idx:
            raw_id_to_idx[raw_id] = len(unique_raws)
            unique_raws.append(raw)

    # Apply subs once to unique raws
    cached_unique = apply_resolved_subs_batch(unique_raws, resolved_subs)

    # Build result with cached versions
    result = []
    for sub_int, ibp_op, shift, raw in indirect_raws:
        cached = cached_unique[raw_id_to_idx[id(raw)]]
        result.append((sub_int, ibp_op, shift, raw, cached))

    return result


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
    """
    import time as _time
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
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            # Apply subs and check
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            if should_filter(cached, target):
                continue
            delta = tuple(seed[i] - target[i] for i in range(7))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))
    t1a_elapsed = _time.time() - t1a

    # Phase 1b: Filter precomputed indirect cache (target-dependent filtering only)
    t1b = _time.time()
    n_indirect_checked = 0
    n_indirect_valid = 0

    for sub_int, ibp_op, shift, raw, cached in indirect_cache:
        n_indirect_checked += 1
        # Check target not in raw (before subs)
        if target in raw and raw[target] != 0:
            continue
        # Check target in cached (after subs)
        if target not in cached or cached[target] == 0:
            continue
        # Check sector filter
        if should_filter(cached, target):
            continue
        seed = tuple(sub_int[i] - shift[i] for i in range(7))
        delta = tuple(seed[i] - target[i] for i in range(7))
        if (ibp_op, delta) not in seen:
            seen.add((ibp_op, delta))
            valid.append((ibp_op, delta))
            n_indirect_valid += 1
    t1b_elapsed = _time.time() - t1b

    total_elapsed = _time.time() - t_start

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
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            delta = tuple(seed[i] - target[i] for i in range(7))
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
                    seed = tuple(sub_int[i] - shift[i] for i in range(7))
                    raw = get_raw_cached(ibp_op, seed)
                    if sub_int in raw and raw[sub_int] != 0:
                        raw_list.append((ibp_op, shift, raw))
            sub_cache[sub_int] = raw_list

        # Now check target against the precomputed raw equations
        for ibp_op, shift, raw in raw_list:
            n_indirect_checked += 1
            if target in raw and raw[target] != 0:
                continue
            seed = tuple(sub_int[i] - shift[i] for i in range(7))
            delta = tuple(seed[i] - target[i] for i in range(7))
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
        self.ibp_t = parse_templates(ibp_path or IBP_PATH)
        self.li_t = parse_templates(li_path or LI_PATH)

        # Build shifts lookup
        self.shifts = {}
        for ibp_op in range(8):
            if ibp_op in self.ibp_t:
                self.shifts[ibp_op] = [s for s, _ in self.ibp_t[ibp_op]]
        for li_idx in self.li_t:
            self.shifts[8 + li_idx] = [s for s, _ in self.li_t[li_idx]]

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
        seed = tuple(target[i] + delta[i] for i in range(7))
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
        seed = tuple(target[i] + delta[i] for i in range(7))
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
