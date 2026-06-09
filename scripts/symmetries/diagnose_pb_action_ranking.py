"""
For the failing pentagon-box target, brute-force find every (ibp_op, delta)
action that REDUCES the max-weight when applied to {target: 1}, and report
the model's rank for each.

If the good actions exist but the model ranks them poorly, that's a model
quality / training issue. If no good actions exist, the search space is
already pathological.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask, max_weight
from beam_search import prepare_batched_input_v5

# Setup
topology = Topology.from_dir(ROOT / "topology_input/pentagonbox")
ibp_env.init_from_topology(topology)
ibp_env.set_prime(1009)
env = ibp_env.IBPEnvironment()

# Load model
ckpt_path = ROOT / "checkpoints/pentagonbox_v2/best_model.pt"
ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
ckpt_args = ckpt.get('args', {})
model = IBPActionClassifier(
    embed_dim=ckpt_args.get('embed_dim', 256),
    n_heads=ckpt_args.get('n_heads', 4),
    n_expr_layers=ckpt_args.get('n_expr_layers', 2),
    n_cross_layers=ckpt_args.get('n_cross_layers', 2),
    n_subs_layers=ckpt_args.get('n_subs_layers', 2),
    prime=1009,
    n_indices=topology.n_indices,
    n_denominators=topology.n_denominators,
    n_ibp_ops=topology.n_actions,
)
sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

target = (1, 0, 1, 0, 1, 1, -1, 0, 0, 0, -1)
target_sector = tuple(get_sector_mask(target))
start_expr = {target: 1}
initial_weight = ibp_env.weight(target)
print(f"target = {target}")
print(f"initial weight = ({initial_weight[0]}, {initial_weight[1]})")
print(f"target_sector = {target_sector}")
print()

# Enumerate all valid actions (subsector filter, like the worker uses)
indirect_cache = env.compute_indirect_cache({}, {})
valid_actions = env.get_valid_actions_with_cache(target, indirect_cache, {}, {},
                                                  filter_mode='subsector')
print(f"Total valid actions (subsector filter): {len(valid_actions)}")
print()

# For each action, apply it to {target: 1} and compute the resulting max_weight
PRIME = ibp_env.PRIME
action_results = []  # (ibp_op, delta, new_max_weight, n_non_masters_new)
for (ibp_op, delta) in valid_actions:
    # seed = target - delta (we want the IBP equation seeded at target+delta_to_target=target, etc.)
    # The "delta" recorded in valid_actions is (seed - target). So seed = target + delta.
    seed = tuple(target[i] + delta[i] for i in range(topology.n_indices))
    raw_eq = ibp_env.get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
    if target not in raw_eq or raw_eq[target] == 0:
        continue
    # Solve for target: raw_eq[target] * target + sum_others c_j * other_j = 0
    # → target = -sum (c_j / c_target) * other_j   (mod PRIME)
    sub = ibp_env.solve_ibp_for(raw_eq, target)
    if sub is None:
        continue
    # Apply substitution to start_expr = {target: 1}: new_expr = sub
    new_expr = dict(sub)
    # Filter to target_sector for max_weight purposes (like beam_search does)
    new_w = max_weight(new_expr, target_sector)
    n_nm = sum(1 for k in new_expr.keys()
                if list(get_sector_mask(k)) == list(target_sector)
                and not ibp_env.is_master(k))
    action_results.append((ibp_op, delta, new_w, n_nm, len(new_expr)))

# Sort by new_max_weight ascending
action_results.sort(key=lambda x: (x[2][0], x[2][1], x[3]))

# How many ACTUALLY reduce weight below initial?
initial_w_tuple = (initial_weight[0], initial_weight[1])
reducing = [r for r in action_results if r[2] < initial_w_tuple]
print(f"Of {len(action_results)} actions applied: {len(reducing)} REDUCE max_weight below {initial_w_tuple}.")
print()

print("Top 10 best-reducing actions (lowest new max_weight, then n_non_masters):")
for ibp_op, delta, new_w, n_nm, n_terms in action_results[:10]:
    print(f"  ibp_op={ibp_op:2d}, delta={delta}, new_max_weight={new_w}, n_non_masters={n_nm}, n_terms={n_terms}")

# Now: get the model's score for each of these actions on the input state
print()
print("=== Running model on the initial state to score all actions ===")
# Prepare input
batch_data = [(start_expr, {}, valid_actions, target_sector, target)]
inputs, n_valid_list = prepare_batched_input_v5(batch_data, device='cpu')
with torch.no_grad():
    logits, _ = model(
        inputs['expr_integrals'], inputs['expr_coeffs'], inputs['expr_mask'],
        inputs['sub_keys'], inputs['sub_repl_ints'], inputs['sub_repl_coeffs'],
        inputs['sub_repl_mask'], inputs['sub_mask'],
        inputs['action_ibp_ops'], inputs['action_deltas'], inputs['action_mask'],
        inputs['sector_mask'], inputs['target_integral'],
    )
# logits is (1, n_valid_actions). Get a ranking.
logits_np = logits[0].cpu().numpy()
n_actions_used = int(inputs['action_mask'][0].sum().item())
logits_np = logits_np[:n_actions_used]
order = (-logits_np).argsort()  # descending
ranking = {idx: rank for rank, idx in enumerate(order)}

# Match valid_actions with reducing actions and report model rank
print(f"Model scored {n_actions_used} actions (= len(valid_actions))")
print()
print("Ranks of the 10 BEST-REDUCING actions:")
print(f"  {'rank':>5}  {'logit':>7}  ibp_op  delta  new_max_w  n_nm")
for ibp_op, delta, new_w, n_nm, _ in action_results[:10]:
    # Find this action's index in the original valid_actions order
    try:
        idx = valid_actions.index((ibp_op, delta))
        rank = ranking[idx]
        print(f"  {rank:>5}  {logits_np[idx]:>+.3f}  {ibp_op:6d}  {delta}  {new_w}  {n_nm}")
    except ValueError:
        pass
