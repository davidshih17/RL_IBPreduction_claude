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

def set_prime(p):
    """Set the prime for modular arithmetic."""
    global PRIME
    PRIME = p
    print(f"IBP environment using PRIME = {PRIME}")

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
    d = 41
    m1, m2, m3 = 1, 31, 47
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
    """
    # Check if it's one of the 16 paper masters
    if i in PAPER_MASTERS:
        return True

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
