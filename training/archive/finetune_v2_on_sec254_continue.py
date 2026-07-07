#!/usr/bin/env python3
"""Continue fine-tuning from finetune_v2_on_sec254 epoch 5 checkpoint
for another 25 epochs at lr=1e-5."""
import sys, os, time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
from torch.utils.data import DataLoader

from sailir.topology import Topology
from sailir.classifier import IBPActionClassifier
from train_classifier import PackedDatasetV5, make_collate_fn, train_epoch, evaluate

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
TOPO_DIR = f'{BASE}/topology_input/pentagonbox'
RESUME_CKPT = f'{BASE}/checkpoints/pentagonbox_sec254_finetune_v1/epoch_5.pt'
DATA_DIR = f'{BASE}/data/pentagonbox_sec254_packed'
OUT_DIR = f'{BASE}/checkpoints/pentagonbox_sec254_finetune_v1'

START_EPOCH = 6
END_EPOCH = 30      # train epochs 6..30 (25 more)
BATCH_SIZE = 128
LR = 1e-5
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda'

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Continuing fine-tune from {RESUME_CKPT}", flush=True)
print(f"Epochs {START_EPOCH}..{END_EPOCH}, lr={LR}", flush=True)

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

ckpt = torch.load(RESUME_CKPT, weights_only=False, map_location=DEVICE)
model = IBPActionClassifier(
    embed_dim=256, n_heads=4,
    n_expr_layers=2, n_cross_layers=2, n_subs_layers=2,
    prime=1009,
    n_indices=n_indices, n_denominators=n_denominators, n_ibp_ops=n_actions,
).to(DEVICE)

sd = ckpt['model_state_dict']
if any(k.startswith('module.') for k in sd):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
print(f"Loaded weights from epoch_5.pt (val top1 was {ckpt.get('val_top1', '?')})", flush=True)

if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Baseline check
val_m = evaluate(model, val_loader, DEVICE)
print(f"Starting val: loss={val_m['loss']:.4f} top1={val_m['top1_acc']:.4f} top5={val_m['top5_acc']:.4f}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for epoch in range(START_EPOCH, END_EPOCH + 1):
    t0 = time.time()
    train_m = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
    val_m = evaluate(model, val_loader, DEVICE)
    print(f"Epoch {epoch}: "
          f"train loss={train_m['loss']:.4f} top1={train_m['top1_acc']:.4f} top5={train_m['top5_acc']:.4f} | "
          f"val loss={val_m['loss']:.4f} top1={val_m['top1_acc']:.4f} top5={val_m['top5_acc']:.4f} | "
          f"{time.time()-t0:.0f}s", flush=True)

    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch, 'model_state_dict': state_dict,
        'val_loss': val_m['loss'], 'val_top1': val_m['top1_acc'],
    }, f'{OUT_DIR}/epoch_{epoch}.pt')

print("Done.", flush=True)
