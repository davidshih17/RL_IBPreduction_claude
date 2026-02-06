#!/usr/bin/env python3
"""
Generate classifier training data for IBP reduction across ALL sectors.

Key differences from single-sector version:
1. Randomly samples from sectors (with paper masters or corner integrals)
2. Uses sector-parameterized functions for is_in_sector, is_higher_sector, is_master
3. Includes sector_id in output for model conditioning

Each sample contains:
- sector_id: binary encoding of the sector being reduced
- Current expression
- Previous substitutions
- Valid action space
- Chosen action (the label)

Output: JSONL file with one sample per reduction step.
"""

import re
import random
import json
import argparse
from pathlib import Path

# Global prime - set by command line argument
PRIME = None

# =============================================================================
# 16 Master Integrals from arXiv:2502.05121 (equation 2.5)
# =============================================================================
PAPER_MASTERS = {
    14: [(0, 1, 1, 1, 0, 0, 0)],                    # sector 14
    21: [(1, 0, 1, 0, 1, 0, 0)],                    # sector 21
    27: [(1, 1, 0, 1, 1, 0, 0)],                    # sector 27
    28: [(0, 0, 1, 1, 1, 0, 0)],                    # sector 28
    29: [(1, 0, 1, 1, 1, 0, 0), (1, -1, 1, 1, 1, 0, 0)],  # sector 29 (2 masters)
    30: [(0, 1, 1, 1, 1, 0, 0), (-1, 1, 1, 1, 1, 0, 0)],  # sector 30 (2 masters)
    31: [(1, 1, 1, 1, 1, 0, 0)],                    # sector 31
    37: [(1, 0, 1, 0, 0, 1, 0)],                    # sector 37
    43: [(1, 1, 0, 1, 0, 1, 0)],                    # sector 43
    53: [(1, 0, 1, 0, 1, 1, 0), (1, -1, 1, 0, 1, 1, 0)],  # sector 53 (2 masters)
    57: [(1, 0, 0, 1, 1, 1, 0)],                    # sector 57
    59: [(1, 1, 0, 1, 1, 1, 0)],                    # sector 59
    61: [(1, 0, 1, 1, 1, 1, 0)],                    # sector 61
}

# Sectors that have paper masters
SECTORS_WITH_PAPER_MASTERS = set(PAPER_MASTERS.keys())


def get_sector_id(integral):
    """
    Compute sector ID from integral indices.
    Sector ID = binary encoding of which propagators (indices 0-5) have power >= 1.
    Index 6 is the ISP and doesn't contribute to sector ID.
    """
    sector_id = 0
    for i in range(6):  # Only first 6 indices (propagators)
        if integral[i] >= 1:
            sector_id += (1 << i)
    return sector_id


def get_sector_mask(sector_id):
    """
    Get the sector mask (which propagators must be >= 1).
    Returns tuple of 6 booleans.
    """
    return tuple((sector_id >> i) & 1 for i in range(6))


def get_corner_integral(sector_id):
    """
    Get the corner integral for a sector.
    Corner = (mask[0], mask[1], ..., mask[5], 0) where mask[i] = 1 if bit i is set.
    """
    mask = get_sector_mask(sector_id)
    return tuple(mask) + (0,)


def get_masters_for_sector(sector_id):
    """
    Get list of master integrals for a sector.
    Uses paper masters if available, otherwise corner integral.
    """
    if sector_id in PAPER_MASTERS:
        return [tuple(m) for m in PAPER_MASTERS[sector_id]]
    else:
        return [get_corner_integral(sector_id)]


def is_in_sector(integral, sector_id):
    """
    Check if integral belongs to the given sector (or a subsector).
    An integral is in sector S if:
    - For each propagator i where S requires power >= 1, integral[i] >= 1
    """
    mask = get_sector_mask(sector_id)
    for i in range(6):
        if mask[i] and integral[i] < 1:
            return False
    return True


def is_higher_sector(integral, sector_id):
    """
    Check if integral is in a strictly higher sector than sector_id.
    A sector T is higher than S if T > S (as binary numbers), meaning T has
    additional propagators with power >= 1 beyond those required by S.
    """
    if not is_in_sector(integral, sector_id):
        return False

    mask = get_sector_mask(sector_id)
    for i in range(6):
        # If this propagator is NOT required by sector but integral has it >= 1
        if not mask[i] and integral[i] >= 1:
            return True
    # Also check ISP (index 6) - if ISP has power >= 1, it's a higher sector
    if integral[6] >= 1:
        return True
    return False


def is_master_for_sector(integral, sector_id):
    """Check if integral is a master for the given sector."""
    masters = get_masters_for_sector(sector_id)
    return tuple(integral) in masters


def weight(integral):
    """Compute weight for ordering integrals (higher weight = reduce first)."""
    return (
        sum(max(0, x) for x in integral),
        -sum(min(0, x) for x in integral),
        tuple(abs(x) for x in integral)
    )


def filter_sector_only(expr, sector_id):
    """Filter to integrals in the given sector, excluding higher sectors."""
    return {k: v for k, v in expr.items()
            if is_in_sector(k, sector_id) and not is_higher_sector(k, sector_id)}


# =============================================================================
# Modular arithmetic
# =============================================================================

def mod_inverse(a, p=None):
    if p is None:
        p = PRIME
    return pow(a % p, p - 2, p)


def solve_ibp_for(ibp, integral):
    if integral not in ibp or ibp[integral] == 0:
        return None
    neg_inv = (PRIME - mod_inverse(ibp[integral])) % PRIME
    return {k: (neg_inv * v) % PRIME for k, v in ibp.items() if k != integral}


def apply_substitution(expr, sub_int, sol):
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
    result = dict(expr)
    for sub_int, sol in subs.items():
        result = apply_substitution(result, sub_int, sol)
    return result


# =============================================================================
# IBP/LI template handling
# =============================================================================

def parse_templates(path):
    templates = {}
    current_idx, terms = None, []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_idx is not None and terms:
                    templates[current_idx] = terms
                current_idx, terms = None, []
                continue
            if '→' in line:
                line = line.split('→', 1)[1]
            match = re.match(r'trianglebox\[([^\]]+)\]\*\(([^)]+)\)', line)
            if match:
                shift = tuple(int(x.strip()) for x in match.group(1).split(','))
                if current_idx is None:
                    current_idx = len(templates)
                terms.append((shift, match.group(2)))
    if current_idx is not None and terms:
        templates[current_idx] = terms
    return templates


def evaluate_coefficient(coeff_str, seed):
    a0, a1, a2, a3, a4, a5, a6 = seed
    d, m1, m2, m3 = 41, 1, 31, 47
    try:
        return eval(coeff_str.replace('^', '**')) % PRIME
    except:
        return 0


def get_raw_equation(ibp_t, li_t, ibp_op, seed):
    if ibp_op >= 8:
        template = li_t.get(ibp_op - 8, [])
    else:
        template = ibp_t.get(ibp_op, [])
    eq = {}
    for shift, coeff_str in template:
        coeff = evaluate_coefficient(coeff_str, seed)
        if coeff:
            eq[tuple(seed[i] + shift[i] for i in range(7))] = coeff
    return eq


# =============================================================================
# Scrambling (reverse reduction)
# =============================================================================

def action_introduces_higher_sector(cached_eq, sector_id):
    """Check if an IBP equation introduces higher sector integrals."""
    for integral in cached_eq.keys():
        if is_in_sector(integral, sector_id) and is_higher_sector(integral, sector_id):
            return True
    return False


def get_integral_props(integral):
    """Get the set of propagator indices where integral has power >= 1."""
    return {i for i in range(6) if integral[i] >= 1}


def is_lateral_sector(integral, sector_id):
    """Check if integral is in a lateral sector to sector_id.

    Lateral = integral's propagator set is neither a subset nor a superset
    of sector_id's propagator set.

    Example: sector 53 has props {0,2,4,5}
             sector 54 has props {1,2,4,5} - lateral (has 1, missing 0)
             sector 52 has props {2,4,5} - NOT lateral (subset)
             sector 55 has props {0,1,2,4,5} - NOT lateral (superset)
    """
    int_props = get_integral_props(integral)
    target_props = {i for i in range(6) if (sector_id >> i) & 1}
    is_subset = int_props <= target_props
    is_superset = int_props >= target_props
    return not is_subset and not is_superset


def action_introduces_outside_sector(cached_eq, sector_id):
    """Check if an IBP equation introduces integrals outside the target sector's cone.

    An integral is "outside" if it's in a higher sector or a lateral sector.
    Subsectors are OK (their propagator set is a subset of target's).

    Returns True if any integral in cached_eq is:
    - In a higher sector (supersector of target) - has extra propagators
    - In a lateral sector - has some extra propagators AND missing some required ones
    """
    for integral in cached_eq.keys():
        if is_higher_sector(integral, sector_id):
            return True
        if is_lateral_sector(integral, sector_id):
            return True
    return False


def find_candidates(ibp_t, li_t, expr, num_ops, sector_id):
    """Find candidate (ibp_op, seed) pairs for scrambling."""
    candidates = []
    for integral in expr:
        if is_in_sector(integral, sector_id) and not is_higher_sector(integral, sector_id):
            for ibp_op in range(num_ops):
                raw = get_raw_equation(ibp_t, li_t, ibp_op, integral)
                if raw and any(k in expr and is_in_sector(k, sector_id) for k in raw):
                    candidates.append((ibp_op, integral))
    return candidates


def scramble(start, ibp_t, li_t, num_ops, n_steps, sector_id, filter_lateral=False):
    """Scramble expression using only actions that don't introduce higher/lateral sectors."""
    expr = dict(start)
    used_ibps = []

    for step in range(n_steps):
        candidates = find_candidates(ibp_t, li_t, expr, num_ops, sector_id)
        if not candidates:
            break

        random.shuffle(candidates)
        found_good_candidate = False

        for ibp_op, seed in candidates:
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)

            # Check if this IBP introduces higher sector (and optionally lateral sectors)
            if filter_lateral:
                if action_introduces_outside_sector(raw, sector_id):
                    continue
            else:
                if action_introduces_higher_sector(raw, sector_id):
                    continue

            top_only = [k for k in raw
                       if is_in_sector(k, sector_id) and not is_higher_sector(k, sector_id)]
            if not top_only:
                continue

            elim = random.choice(top_only)
            sol = solve_ibp_for(raw, elim)
            if sol is None:
                continue

            if elim in expr:
                expr = apply_substitution(expr, elim, sol)
            else:
                for k, v in raw.items():
                    expr[k] = (expr.get(k, 0) + v) % PRIME
                expr = {k: v for k, v in expr.items() if v != 0}

            used_ibps.append((ibp_op, seed))
            found_good_candidate = True
            break

        if not found_good_candidate:
            break

    return expr, used_ibps


# =============================================================================
# Action enumeration
# =============================================================================

def enumerate_valid_actions(target, subs, ibp_t, li_t, shifts, sector_id,
                           filter_higher=True, filter_lateral=False):
    """
    Enumerate all valid (ibp_op, delta) pairs that can eliminate target.
    Returns list of (ibp_op, delta) where delta = seed - target.

    Args:
        filter_higher: Filter out actions introducing higher sector integrals
        filter_lateral: Also filter out actions introducing lateral sector integrals
    """
    valid = []
    seen = set()

    # Direct actions: target directly in raw_ibp
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(7))
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_all_substitutions(raw, subs)
            if target not in cached or cached[target] == 0:
                continue
            # Filter based on options
            if filter_higher:
                if filter_lateral:
                    if action_introduces_outside_sector(cached, sector_id):
                        continue
                else:
                    if action_introduces_higher_sector(cached, sector_id):
                        continue
            delta = tuple(seed[i] - target[i] for i in range(7))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta))

    # Indirect actions: target via substitution chain
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
                # Filter based on options
                if filter_higher:
                    if filter_lateral:
                        if action_introduces_outside_sector(cached, sector_id):
                            continue
                    else:
                        if action_introduces_higher_sector(cached, sector_id):
                            continue
                delta = tuple(seed[i] - target[i] for i in range(7))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid


# =============================================================================
# JSON serialization
# =============================================================================

def expr_to_json(expr):
    return [[list(k), v] for k, v in sorted(expr.items())]


def subs_to_json(subs):
    return [[list(k), [[list(ki), vi] for ki, vi in sorted(v.items())]]
            for k, v in sorted(subs.items())]


# =============================================================================
# Main
# =============================================================================

def get_all_valid_sectors():
    """
    Get list of all 63 non-trivial sectors (sector_id 1-63).
    """
    return list(range(1, 64))


def main():
    global PRIME

    parser = argparse.ArgumentParser(
        description='Generate multi-sector IBP reduction training data')
    parser.add_argument('--n_scrambles', type=int, default=1000,
                        help='Number of scramble trajectories to generate')
    parser.add_argument('--min_steps', type=int, default=5,
                        help='Minimum scramble steps')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum scramble steps')
    parser.add_argument('--output', type=str,
                        default='data/multisector_training_data.jsonl',
                        help='Output file path')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting random seed')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--ibp_path', type=str,
                        default='scripts/data_gen/IBP',
                        help='Path to IBP templates')
    parser.add_argument('--li_path', type=str,
                        default='scripts/data_gen/LI',
                        help='Path to LI templates')
    parser.add_argument('--filter_lateral', action='store_true',
                        help='Also filter out actions that introduce lateral sector integrals')
    args = parser.parse_args()

    PRIME = args.prime

    print(f"=" * 70, flush=True)
    print(f"Multi-Sector IBP Training Data Generator", flush=True)
    print(f"=" * 70, flush=True)
    print(f"PRIME = {PRIME}", flush=True)
    print(f"Scrambles: {args.n_scrambles}", flush=True)
    print(f"Steps: {args.min_steps}-{args.max_steps}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Filter lateral sectors: {args.filter_lateral}", flush=True)

    # Always use all 63 sectors
    sector_list = get_all_valid_sectors()
    print(f"Using all {len(sector_list)} sectors (1-63)", flush=True)

    # Load IBP/LI templates
    ibp_t = parse_templates(args.ibp_path)
    li_t = parse_templates(args.li_path)
    num_ops = len(ibp_t) + len(li_t)
    print(f"Loaded {len(ibp_t)} IBP templates, {len(li_t)} LI templates", flush=True)

    # Build shifts lookup
    shifts = {}
    for ibp_op in range(8):
        if ibp_op in ibp_t:
            shifts[ibp_op] = [s for s, _ in ibp_t[ibp_op]]
    for li_idx in li_t:
        shifts[8 + li_idx] = [s for s, _ in li_t[li_idx]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    successful_scrambles = 0
    failed_scrambles = 0
    skipped_vanishing = 0  # Sectors with vanishing corners
    sector_counts = {}  # Track samples per sector

    with open(args.output, 'w') as f:
        for scramble_idx in range(args.n_scrambles):
            seed_value = args.start_seed + scramble_idx * 1000
            random.seed(seed_value)

            # Sample a sector
            sector_id = random.choice(sector_list)
            masters = get_masters_for_sector(sector_id)

            # Create random linear combination of masters
            start = {}
            master_coeffs = []
            for m in masters:
                c = random.randint(1, PRIME - 1)
                start[m] = c
                master_coeffs.append(c)

            # Scramble
            n_steps = random.randint(args.min_steps, args.max_steps)
            scrambled, used_ibps = scramble(start, ibp_t, li_t, num_ops, n_steps, sector_id,
                                           filter_lateral=args.filter_lateral)

            if not used_ibps:
                # Check if this is a vanishing corner (expression became empty or trivial)
                # This happens when the corner integral is zero by IBP identities
                skipped_vanishing += 1
                print(f"SKIPPED_VANISHING: scramble={scramble_idx}, sector={sector_id}, "
                      f"masters={masters}, reason=no_valid_scramble_actions", flush=True)
                continue

            # Unscramble and collect training samples
            expr = dict(scrambled)
            subs = {}
            used_set = set()
            scramble_samples = []
            success = True
            failure_reason = None

            for iteration in range(500):
                sector_only = filter_sector_only(expr, sector_id)
                non_masters = {k: v for k, v in sector_only.items()
                              if not is_master_for_sector(k, sector_id)}

                if not non_masters:
                    # Success - verify coefficients
                    final = {k: v for k, v in sector_only.items()
                            if is_master_for_sector(k, sector_id)}
                    expected_coeffs = {m: c for m, c in zip(masters, master_coeffs)}
                    if final != expected_coeffs:
                        success = False
                        failure_reason = f"coeff_mismatch: final={final}, expected={expected_coeffs}"
                    break

                target = sorted(non_masters.keys(),
                    key=lambda x: (-weight(x)[0], -weight(x)[1], weight(x)[2]))[0]

                # Find the used_ibps action that works
                found_action = None
                for idx, (ibp_op, seed) in enumerate(used_ibps):
                    if idx in used_set:
                        continue
                    raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)
                    cached = apply_all_substitutions(raw, subs)
                    if target in cached and cached[target] != 0:
                        found_action = (idx, ibp_op, seed)
                        break

                if found_action is None:
                    success = False
                    failure_reason = f"no_action_found: iteration={iteration}, target={target}"
                    break

                idx, chosen_ibp_op, chosen_seed = found_action
                chosen_delta = tuple(chosen_seed[i] - target[i] for i in range(7))

                # Enumerate valid actions
                valid_actions = enumerate_valid_actions(
                    target, subs, ibp_t, li_t, shifts, sector_id,
                    filter_lateral=args.filter_lateral)

                # Verify chosen action is in valid actions
                if (chosen_ibp_op, chosen_delta) not in valid_actions:
                    success = False
                    failure_reason = f"action_not_valid: iteration={iteration}, action=({chosen_ibp_op}, {chosen_delta})"
                    break

                chosen_idx = valid_actions.index((chosen_ibp_op, chosen_delta))

                # Create training sample with sector_id
                sample = {
                    'scramble_id': scramble_idx,
                    'sector_id': sector_id,  # NEW: include sector for model conditioning
                    'sector_mask': list(get_sector_mask(sector_id)),  # NEW: 6-bit mask
                    'step': iteration,
                    'target': list(target),
                    'target_weight': weight(target)[:2],
                    'expr': expr_to_json(filter_sector_only(expr, sector_id)),
                    'subs': subs_to_json(subs),
                    'valid_actions': [[ibp_op, list(delta)] for ibp_op, delta in valid_actions],
                    'num_valid_actions': len(valid_actions),
                    'chosen_action': [chosen_ibp_op, list(chosen_delta)],
                    'chosen_action_idx': chosen_idx
                }
                scramble_samples.append(sample)

                # Apply the action
                raw = get_raw_equation(ibp_t, li_t, chosen_ibp_op, chosen_seed)
                cached = apply_all_substitutions(raw, subs)
                sol = solve_ibp_for(cached, target)

                used_set.add(idx)
                subs[target] = sol
                expr = apply_substitution(expr, target, sol)
            else:
                success = False
                failure_reason = "max_iterations_reached"

            if success:
                for sample in scramble_samples:
                    f.write(json.dumps(sample) + '\n')
                total_samples += len(scramble_samples)
                successful_scrambles += 1
                sector_counts[sector_id] = sector_counts.get(sector_id, 0) + len(scramble_samples)
            else:
                # Coefficient mismatch typically indicates vanishing corner
                if failure_reason and failure_reason.startswith("coeff_mismatch"):
                    skipped_vanishing += 1
                    print(f"SKIPPED_VANISHING: scramble={scramble_idx}, sector={sector_id}, "
                          f"masters={masters}, reason={failure_reason}", flush=True)
                else:
                    failed_scrambles += 1
                    print(f"FAILED: scramble={scramble_idx}, sector={sector_id}, "
                          f"masters={masters}, reason={failure_reason}", flush=True)

            if (scramble_idx + 1) % 100 == 0:
                print(f"Progress: {scramble_idx + 1}/{args.n_scrambles}, "
                      f"samples={total_samples}, success={successful_scrambles}, "
                      f"skipped_vanishing={skipped_vanishing}, fail={failed_scrambles}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"Done! {total_samples} training samples from {successful_scrambles} scrambles", flush=True)
    print(f"Skipped (vanishing corners): {skipped_vanishing}", flush=True)
    print(f"Failed scrambles: {failed_scrambles}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"\nSamples per sector:", flush=True)
    for sid in sorted(sector_counts.keys()):
        print(f"  Sector {sid}: {sector_counts[sid]} samples", flush=True)


if __name__ == "__main__":
    main()
