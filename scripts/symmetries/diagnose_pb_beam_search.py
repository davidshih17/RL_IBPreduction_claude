"""
Run beam_search locally on the failing integral with full verbose output,
to see exactly what it's doing.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reduction"))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.classifier import IBPActionClassifier
from beam_search import beam_search
from beam_search_utils import get_sector_mask, max_weight, get_non_masters

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
# Handle DataParallel module prefix if present
sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

target = (1, 0, 1, 0, 1, 1, -1, 0, 0, 0, -1)
start_expr = {target: 1}
target_sector = tuple(get_sector_mask(target))
print(f"target = {target}")
print(f"target_sector = {target_sector}")
print(f"start_expr non-masters in target sector = {get_non_masters(start_expr, target_sector)}")
print(f"max_weight in target sector = {max_weight(start_expr, target_sector)}")
print()
print("Calling beam_search ...")
solution, final_beam, best_weight = beam_search(
    env, model, start_expr,
    beam_width=20,
    max_steps=20,           # small max_steps to keep diagnostic quick
    device='cpu',
    verbose=True,
    target_sector=target_sector,
    filter_mode='subsector',
    use_resolved_subs=True,
    n_workers=1,
    prime=1009,
    patience=None,
    stop_on_weight_improvement=True,
)
print()
print(f"=== beam_search returned ===")
print(f"  solution: {solution is not None}")
print(f"  final_beam size: {len(final_beam)}")
print(f"  best_weight: {best_weight}")
if final_beam:
    for i, s in enumerate(final_beam[:3]):
        w = max_weight(s.expr, target_sector)
        nms = get_non_masters(s.expr, target_sector)
        print(f"  beam[{i}]: weight={w}, n_non_masters={len(nms)}, path_len={len(s.path)}")
        print(f"    expr keys: {list(s.expr.keys())[:5]}")
