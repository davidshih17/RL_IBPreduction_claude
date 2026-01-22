#!/usr/bin/env python3
"""
Generate classifier training data for IBP reduction with configurable prime.

Using smaller primes (e.g., p=1009) instead of 2147483647 makes coefficients
more learnable. Fractions like -2/47 become small integers like 64 instead
of huge values like 639675980.

Each sample contains:
- Current expression
- Previous substitutions
- Valid action space: (delta, ibp_op) pairs where delta = seed - target
- Which action was chosen (the label)

Output: JSONL file with one sample per reduction step.
"""

import re
import random
import json
import argparse
from pathlib import Path

# Global prime - set by command line argument
PRIME = None

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

def is_top_sector(i): return i[0] >= 1 and i[2] >= 1 and i[4] >= 1 and i[5] >= 1
def is_higher_sector(i): return is_top_sector(i) and (i[1] >= 1 or i[3] >= 1 or i[6] >= 1)
def is_master(i): return i in [(1, 0, 1, 0, 1, 1, 0), (1, -1, 1, 0, 1, 1, 0)]
def weight(i):
    return (sum(max(0, x) for x in i), -sum(min(0, x) for x in i), tuple(abs(x) for x in i))
def filter_top_sector_only(expr):
    return {k: v for k, v in expr.items() if is_top_sector(k) and not is_higher_sector(k)}

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
            if '→' in line: line = line.split('→', 1)[1]
            match = re.match(r'trianglebox\[([^\]]+)\]\*\(([^)]+)\)', line)
            if match:
                shift = tuple(int(x.strip()) for x in match.group(1).split(','))
                if current_idx is None: current_idx = len(templates)
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

def find_candidates(ibp_t, li_t, expr, num_ops):
    candidates = []
    for integral in expr:
        if is_top_sector(integral) and not is_higher_sector(integral):
            for ibp_op in range(num_ops):
                raw = get_raw_equation(ibp_t, li_t, ibp_op, integral)
                if raw and any(k in expr and is_top_sector(k) for k in raw):
                    candidates.append((ibp_op, integral))
    return candidates

def scramble(start, ibp_t, li_t, num_ops, n_steps):
    """Scramble expression using only actions that don't introduce higher sectors."""
    expr = dict(start)
    used_ibps = []
    for step in range(n_steps):
        candidates = find_candidates(ibp_t, li_t, expr, num_ops)
        if not candidates:
            break

        # Shuffle candidates and try to find one that doesn't introduce higher sector
        random.shuffle(candidates)
        found_good_candidate = False

        for ibp_op, seed in candidates:
            raw = get_raw_equation(ibp_t, li_t, ibp_op, seed)

            # Check if this IBP introduces higher sector
            if action_introduces_higher_sector(raw):
                continue

            top_only = [k for k in raw if is_top_sector(k) and not is_higher_sector(k)]
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

def action_introduces_higher_sector(cached_eq):
    """Check if an IBP equation (after subs) introduces higher sector integrals."""
    for integral in cached_eq.keys():
        if is_top_sector(integral) and is_higher_sector(integral):
            return True
    return False


def enumerate_valid_actions(target, subs, ibp_t, li_t, shifts, filter_higher_sector=True):
    """
    Enumerate all valid (ibp_op, seed) pairs that can eliminate target.
    Returns list of (ibp_op, delta) where delta = seed - target.

    Args:
        filter_higher_sector: If True (default), exclude actions that introduce higher sector integrals.
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
            # Filter out actions that introduce higher sector integrals
            if filter_higher_sector and action_introduces_higher_sector(cached):
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
                    continue  # Skip direct (already enumerated)
                cached = apply_all_substitutions(raw, subs)
                if target not in cached or cached[target] == 0:
                    continue
                # Filter out actions that introduce higher sector integrals
                if filter_higher_sector and action_introduces_higher_sector(cached):
                    continue
                delta = tuple(seed[i] - target[i] for i in range(7))
                if (ibp_op, delta) not in seen:
                    seen.add((ibp_op, delta))
                    valid.append((ibp_op, delta))

    return valid

def expr_to_json(expr):
    """Convert expression to JSON-serializable format."""
    return [[list(k), v] for k, v in sorted(expr.items())]

def subs_to_json(subs):
    """Convert substitutions to JSON-serializable format."""
    return [[list(k), [[list(ki), vi] for ki, vi in sorted(v.items())]] for k, v in sorted(subs.items())]

def main():
    global PRIME

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_scrambles', type=int, default=1000)
    parser.add_argument('--min_steps', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--output', type=str, default='/home/shih/work/IBPreduction/data/classifier_training_data_p1009.jsonl')
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic (default: 1009, original: 2147483647)')
    args = parser.parse_args()

    PRIME = args.prime

    print(f"Generating classifier training data with PRIME={PRIME}", flush=True)
    print(f"Scrambles: {args.n_scrambles}", flush=True)
    print(f"Output: {args.output}", flush=True)

    ibp_t = parse_templates('/home/shih/work/IBPreduction/scripts/data_gen/IBP')
    li_t = parse_templates('/home/shih/work/IBPreduction/scripts/data_gen/LI')
    num_ops = len(ibp_t) + len(li_t)

    # Build shifts lookup
    shifts = {}
    for ibp_op in range(8):
        if ibp_op in ibp_t:
            shifts[ibp_op] = [s for s, _ in ibp_t[ibp_op]]
    for li_idx in li_t:
        shifts[8 + li_idx] = [s for s, _ in li_t[li_idx]]

    master1 = (1, 0, 1, 0, 1, 1, 0)
    master2 = (1, -1, 1, 0, 1, 1, 0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    successful_scrambles = 0
    failed_scrambles = 0

    with open(args.output, 'w') as f:
        for scramble_idx in range(args.n_scrambles):
            seed_value = args.start_seed + scramble_idx * 1000
            random.seed(seed_value)
            c1 = random.randint(1, PRIME - 1)
            c2 = random.randint(1, PRIME - 1)
            start = {master1: c1, master2: c2}

            n_steps = random.randint(args.min_steps, args.max_steps)
            scrambled, used_ibps = scramble(start, ibp_t, li_t, num_ops, n_steps)

            if not used_ibps:
                failed_scrambles += 1
                continue

            # Unscramble and collect training samples
            expr = dict(scrambled)
            subs = {}
            used_set = set()
            scramble_samples = []
            success = True

            for iteration in range(500):
                top_only = filter_top_sector_only(expr)
                non_masters = {k: v for k, v in top_only.items() if not is_master(k)}

                if not non_masters:
                    # Success - verify coefficients
                    final = {k: v for k, v in top_only.items() if is_master(k)}
                    if final.get(master1, 0) != c1 or final.get(master2, 0) != c2:
                        success = False
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
                    break

                idx, chosen_ibp_op, chosen_seed = found_action
                chosen_delta = tuple(chosen_seed[i] - target[i] for i in range(7))

                # Enumerate valid actions
                valid_actions = enumerate_valid_actions(target, subs, ibp_t, li_t, shifts)

                # Verify chosen action is in valid actions
                if (chosen_ibp_op, chosen_delta) not in valid_actions:
                    print(f"WARNING: Chosen action not in valid actions! scramble={scramble_idx}, iter={iteration}", flush=True)
                    success = False
                    break

                # Find index of chosen action in valid_actions
                chosen_idx = valid_actions.index((chosen_ibp_op, chosen_delta))

                # Create training sample
                sample = {
                    'scramble_id': scramble_idx,
                    'step': iteration,
                    'target': list(target),
                    'target_weight': weight(target)[:2],
                    'expr': expr_to_json(filter_top_sector_only(expr)),
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

            if success:
                # Write all samples from this scramble
                for sample in scramble_samples:
                    f.write(json.dumps(sample) + '\n')
                total_samples += len(scramble_samples)
                successful_scrambles += 1
            else:
                failed_scrambles += 1

            if (scramble_idx + 1) % 100 == 0:
                print(f"Progress: {scramble_idx + 1}/{args.n_scrambles}, samples={total_samples}, success={successful_scrambles}, fail={failed_scrambles}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Done! {total_samples} training samples from {successful_scrambles} scrambles", flush=True)
    print(f"Failed scrambles: {failed_scrambles}", flush=True)
    print(f"Output: {args.output}", flush=True)

if __name__ == "__main__":
    main()
