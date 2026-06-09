#!/usr/bin/env python3
"""Fine-tune v2 checkpoint on sec254-only data for a few epochs.

Diagnostic: if val top1 stays near v2's 0.98, the data/loss pipeline is fine
and the sec254-v1 plateau is purely a from-scratch-on-single-sector issue.
If val top1 collapses, there's a real bug we still need to find.
"""
import sys, os, time
from pathlib import Path

# Same import setup as train_classifier.py
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sailir.topology import Topology
from sailir.classifier import IBPActionClassifier
from train_classifier import PackedDatasetV5, make_collate_fn, train_epoch, evaluate

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
TOPO_DIR = f'{BASE}/topology_input/pentagonbox'
V2_CKPT = f'{BASE}/checkpoints/pentagonbox_v2/best_model.pt'
DATA_DIR = f'{BASE}/data/pentagonbox_sec254_packed'
OUT_DIR = f'{BASE}/checkpoints/pentagonbox_sec254_finetune_v1'

EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-5            # small lr for fine-tune
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda'

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70, flush=True)
print(f"Fine-tune v2 on sec254 data", flush=True)
print("=" * 70, flush=True)
print(f"V2 checkpoint:  {V2_CKPT}", flush=True)
print(f"Data dir:       {DATA_DIR}", flush=True)
print(f"Output dir:     {OUT_DIR}", flush=True)
print(f"Epochs:         {EPOCHS}", flush=True)
print(f"LR:             {LR}", flush=True)
print(f"Batch size:     {BATCH_SIZE}", flush=True)
print(flush=True)

topology = Topology.from_dir(TOPO_DIR)
n_indices, n_denominators, n_actions = topology.n_indices, topology.n_denominators, topology.n_actions

print("Loading packed tensors ...", flush=True)
t0 = time.time()
train_data = torch.load(f'{DATA_DIR}/train.pt', weights_only=False)
val_data = torch.load(f'{DATA_DIR}/val.pt', weights_only=False)
print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

train_ds = PackedDatasetV5(train_data)
val_ds = PackedDatasetV5(val_data)
collate = make_collate_fn(n_indices=n_indices, n_denominators=n_denominators)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=4, pin_memory=True)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

# Build model with same hyperparams as v2 (read from v2 ckpt args)
ckpt = torch.load(V2_CKPT, weights_only=False, map_location=DEVICE)
ckpt_args = ckpt.get('args', {})
model = IBPActionClassifier(
    embed_dim=ckpt_args.get('embed_dim', 256),
    n_heads=ckpt_args.get('n_heads', 4),
    n_expr_layers=ckpt_args.get('n_expr_layers', 2),
    n_cross_layers=ckpt_args.get('n_cross_layers', 2),
    n_subs_layers=ckpt_args.get('n_subs_layers', 2),
    prime=1009,
    n_indices=n_indices, n_denominators=n_denominators, n_ibp_ops=n_actions,
).to(DEVICE)

sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
print(f"Loaded v2 weights: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f"DataParallel over {torch.cuda.device_count()} GPUs", flush=True)

# Evaluate baseline (v2 on sec254 val) first
print(f"\n=== Baseline: v2 on sec254 val ===", flush=True)
m = evaluate(model, val_loader, DEVICE)
print(f"  loss={m['loss']:.4f}, top1={m['top1_acc']:.4f}, top5={m['top5_acc']:.4f}", flush=True)

# Fresh optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nFine-tuning for {EPOCHS} epochs ...", flush=True)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_m = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
    val_m = evaluate(model, val_loader, DEVICE)
    print(f"Epoch {epoch}: "
          f"train loss={train_m['loss']:.4f} top1={train_m['top1_acc']:.4f} top5={train_m['top5_acc']:.4f} | "
          f"val loss={val_m['loss']:.4f} top1={val_m['top1_acc']:.4f} top5={val_m['top5_acc']:.4f} | "
          f"{time.time()-t0:.0f}s", flush=True)

    # Save
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'val_loss': val_m['loss'],
        'val_top1': val_m['top1_acc'],
    }, f'{OUT_DIR}/epoch_{epoch}.pt')

print("\nDone.", flush=True)
