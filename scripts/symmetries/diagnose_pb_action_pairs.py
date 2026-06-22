"""
Brute-force search over (action_A, action_B) PAIRS to find any 2-step
trajectory that reduces max_weight below the initial (4, 2) for the
failing pentagon-box target.

If 2-step reductions exist, beam_search with width >=N and at least 2 steps
should find them — and we can quantify how the model ranks the productive
first-step action.
"""
import sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reduction"))

from sailir.topology import Topology
from sailir import ibp_env
from beam_search_utils import get_sector_mask, max_weight

# Setup
topology = Topology.from_dir(ROOT / "topology_input/pentagonbox")
ibp_env.init_from_topology(topology)
ibp_env.set_prime(1009)
env = ibp_env.IBPEnvironment()

target = (1, 0, 1, 0, 1, 1, -1, 0, 0, 0, -1)
target_sector = tuple(get_sector_mask(target))
start_expr = {target: 1}
initial_weight = ibp_env.weight(target)
initial_w_tuple = (initial_weight[0], initial_weight[1])
print(f"target = {target}  initial_w = {initial_w_tuple}")
PRIME = ibp_env.PRIME

# Step 1: enumerate all single-action results
indirect_cache = env.compute_indirect_cache({}, {})
valid_actions_a = env.get_valid_actions_with_cache(target, indirect_cache, {}, {},
                                                    filter_mode='subsector')
print(f"Step-1 valid actions: {len(valid_actions_a)}")

def apply_action(expr, ibp_op, delta_to_target_target):
    """Apply a single (ibp_op, delta) action where delta is (seed - target_eliminated).

    Returns the new expression dict (modular arithmetic), or None if action invalid.
    """
    target_to_kill_list = [k for k, v in expr.items() if v != 0]
    # In beam_search, each step picks ONE target to kill (highest weight). Here we
    # only call with the max-weight integral as the elim target.
    return None  # not used; we implement inline below


# Collect (action_a, expr_after_a, max_w_after_a) for promising step-1 states
print("Computing all step-1 outcomes...")
step1_outcomes = []
for (op_a, delta_a) in valid_actions_a:
    seed = tuple(target[i] + delta_a[i] for i in range(topology.n_indices))
    raw = ibp_env.get_raw_equation(env.ibp_t, env.li_t, op_a, seed)
    if target not in raw or raw[target] == 0:
        continue
    sub = ibp_env.solve_ibp_for(raw, target)
    if sub is None:
        continue
    # Apply: start_expr = {target: 1}, substitute target -> sub
    new_expr = dict(sub)
    new_w = max_weight(new_expr, target_sector)
    step1_outcomes.append((op_a, delta_a, new_expr, new_w))

print(f"  {len(step1_outcomes)} step-1 outcomes")
print(f"  best step-1 max_weight: {min(o[3] for o in step1_outcomes)}")

# Step 2: for each step-1 state, pick its highest-weight non-master target and
# enumerate all step-2 actions on it, then compute final max_weight.
print()
print("Brute-forcing all (action_a, action_b) pairs ...")

reducing_pairs = []
checked = 0
for (op_a, delta_a, expr_a, w_a) in step1_outcomes:
    # Pick step-2 target: highest-weight non-master in expr_a (in target_sector)
    nms_a = {k: v for k, v in expr_a.items()
              if list(get_sector_mask(k)) == list(target_sector)
              and not ibp_env.is_master(k)}
    if not nms_a:
        continue
    target_b = max(nms_a.keys(), key=lambda k: ibp_env.weight(k)[:2])

    # Enumerate step-2 actions for target_b. Use cached env actions.
    cache_b = env.compute_indirect_cache({}, {})
    valid_b = env.get_valid_actions_with_cache(target_b, cache_b, {}, {},
                                                filter_mode='subsector')
    for (op_b, delta_b) in valid_b:
        seed_b = tuple(target_b[i] + delta_b[i] for i in range(topology.n_indices))
        raw_b = ibp_env.get_raw_equation(env.ibp_t, env.li_t, op_b, seed_b)
        if target_b not in raw_b or raw_b[target_b] == 0:
            continue
        sub_b = ibp_env.solve_ibp_for(raw_b, target_b)
        if sub_b is None:
            continue
        # Apply: replace target_b in expr_a with sub_b
        coeff_tb = expr_a[target_b]
        new_expr = {k: v for k, v in expr_a.items() if k != target_b}
        for ik, vk in sub_b.items():
            cv = (coeff_tb * vk) % PRIME
            new_expr[ik] = (new_expr.get(ik, 0) + cv) % PRIME
        # Drop zeros
        new_expr = {k: v for k, v in new_expr.items() if v != 0}
        new_w = max_weight(new_expr, target_sector)
        checked += 1
        if new_w < initial_w_tuple:
            reducing_pairs.append((op_a, delta_a, op_b, delta_b, target_b, new_w))

print(f"  Checked {checked} pairs.")
print(f"  REDUCING pairs (new max_w < {initial_w_tuple}): {len(reducing_pairs)}")
print()

if reducing_pairs:
    # Group by first action
    by_first = Counter((op_a, delta_a) for op_a, delta_a, *_ in reducing_pairs)
    print(f"Distinct successful first-actions: {len(by_first)}")
    print(f"Top first-actions (by # of reducing partners):")
    for (op_a, delta_a), n in sorted(by_first.items(), key=lambda x: -x[1])[:10]:
        # Get the best new_w for this first action
        best = min((p for p in reducing_pairs if p[0] == op_a and p[1] == delta_a),
                   key=lambda p: p[5])
        print(f"  ibp_op={op_a:2d}, delta={delta_a}, #partners={n}, best_new_w={best[5]}")

    print()
    print(f"Top 5 reducing pair examples:")
    for op_a, delta_a, op_b, delta_b, tb, new_w in sorted(reducing_pairs, key=lambda p: p[5])[:5]:
        print(f"  A: ibp_op={op_a}, delta={delta_a}")
        print(f"  B: ibp_op={op_b}, delta={delta_b}  (on target_b={tb})")
        print(f"  --> new max_w = {new_w}")
        print()
else:
    print("NO reducing pairs found. Reduction is not possible in 2 steps.")
