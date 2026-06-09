"""Run v2 (trained on all-sectors) on the sec254 test set. Compare top1/top5
to sec254_v1 (trained only on sec254)."""
import sys
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/train')

import torch
from torch.utils.data import DataLoader
from sailir.topology import Topology
from sailir.classifier import IBPActionClassifier
from train_classifier import PackedDatasetV5, make_collate_fn

device = 'cuda'

topology = Topology.from_dir('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox')

# Load v2 model
ckpt_path = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_v2/best_model.pt'
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
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
).to(device)

sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f"Loaded v2: {sum(p.numel() for p in model.parameters()):,} params")

# Load sec254 test set
test_data = torch.load(
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/data/pentagonbox_sec254_packed/test.pt',
    weights_only=False
)
test_ds = PackedDatasetV5(test_data)
collate = make_collate_fn(n_indices=topology.n_indices, n_denominators=topology.n_denominators)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate, num_workers=2)
print(f"sec254 test set: {len(test_ds)} samples\n")

# Evaluate
total = 0
top1 = 0
top5 = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral']
        )
        # Apply action mask: actions outside the valid set get -inf
        logits = logits.masked_fill(~batch['action_mask'], float('-inf'))
        labels = batch['labels']
        top1_pred = logits.argmax(dim=-1)
        top1 += (top1_pred == labels).sum().item()
        top5_pred = logits.topk(5, dim=-1).indices
        top5 += (top5_pred == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.size(0)

print(f"v2 (trained on all-sectors) evaluated on sec254 test:")
print(f"  top1 = {top1/total:.4f}")
print(f"  top5 = {top5/total:.4f}")
print()
print(f"For comparison:")
print(f"  sec254_v1 (trained only on sec254) val top1 ~ 0.31, top5 ~ 0.84")
