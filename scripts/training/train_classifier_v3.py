"""
Training script for IBP Action Classifier v3.

Uses transformer encoder for expressions - terms attend to each other.

Usage:
    python -u scripts/training/train_classifier_v3.py --epochs 50 --batch_size 32

Run in background with logging:
    nohup python -u scripts/training/train_classifier_v3.py > logs/train_classifier_v3.log 2>&1 &
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from classifier_v3 import IBPActionClassifierV3, collate_samples
from ibp_env import IBPEnvironment


class ClassifierDataset(Dataset):
    """Dataset for classifier training samples."""

    def __init__(self, data_path, max_samples=None):
        self.samples = []
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(samples, max_terms=200, max_subs=50, max_actions=900):
    """Custom collate function using classifier_v3.collate_samples."""
    return collate_samples(samples, max_terms, max_subs, max_actions)


def compute_accuracy(logits, labels, action_mask):
    """Compute top-1 and top-5 accuracy."""
    preds = logits.argmax(dim=-1)
    top1_correct = (preds == labels).float().sum()

    # Top-5: check if label is in top 5 predictions
    _, top5_indices = logits.topk(min(5, logits.size(1)), dim=-1)
    top5_correct = (top5_indices == labels.unsqueeze(1)).any(dim=1).float().sum()

    return top1_correct, top5_correct


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass
        logits, probs = model(
            batch['expr_integrals'],
            batch['expr_coeffs'],
            batch['expr_mask'],
            batch['sub_integrals'],
            batch['sub_mask'],
            batch['action_ibp_ops'],
            batch['action_deltas'],
            batch['action_mask']
        )

        # Loss
        loss = F.cross_entropy(logits, batch['labels'])

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        batch_size = batch['labels'].size(0)
        total_loss += loss.item() * batch_size
        top1, top5 = compute_accuracy(logits, batch['labels'], batch['action_mask'])
        total_top1 += top1.item()
        total_top5 += top5.item()
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} batch {batch_idx+1}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, top1={top1.item()/batch_size:.3f}", flush=True)

    return {
        'loss': total_loss / total_samples,
        'top1_acc': total_top1 / total_samples,
        'top5_acc': total_top5 / total_samples
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits, probs = model(
            batch['expr_integrals'],
            batch['expr_coeffs'],
            batch['expr_mask'],
            batch['sub_integrals'],
            batch['sub_mask'],
            batch['action_ibp_ops'],
            batch['action_deltas'],
            batch['action_mask']
        )

        loss = F.cross_entropy(logits, batch['labels'])

        batch_size = batch['labels'].size(0)
        total_loss += loss.item() * batch_size
        top1, top5 = compute_accuracy(logits, batch['labels'], batch['action_mask'])
        total_top1 += top1.item()
        total_top5 += top5.item()
        total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'top1_acc': total_top1 / total_samples,
        'top5_acc': total_top5 / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Train IBP Action Classifier v3')
    parser.add_argument('--data', type=str,
                        default='/home/shih/work/IBPreduction/data/classifier_training_data_p1009.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default='/home/shih/work/IBPreduction/checkpoints/classifier_v3_p1009')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_expr_layers', type=int, default=2,
                        help='Number of transformer layers for expression encoding')
    parser.add_argument('--n_cross_layers', type=int, default=2,
                        help='Number of cross-attention layers for action scoring')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit samples for debugging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic (default: 1009)')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("IBP Action Classifier v3 Training", flush=True)
    print("=" * 70, flush=True)
    print(f"Config:", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...", flush=True)
    full_dataset = ClassifierDataset(args.data, max_samples=args.max_samples)

    # Split into train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    # Create datasets (wrap in simple list-based dataset)
    class ListDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_dataset = ListDataset(train_samples)
    val_dataset = ListDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Set prime for IBP environment
    from ibp_env import set_prime
    set_prime(args.prime)

    # Create IBP environment for end-to-end evaluation
    print("\nLoading IBP environment for end-to-end evaluation...", flush=True)
    ibp_env = IBPEnvironment()
    print(f"Loaded {len(ibp_env.ibp_t)} IBP templates, {len(ibp_env.li_t)} LI templates", flush=True)

    # Create model
    print(f"\nCreating model (embed_dim={args.embed_dim}, n_heads={args.n_heads}, "
          f"n_expr_layers={args.n_expr_layers}, n_cross_layers={args.n_cross_layers}, "
          f"prime={args.prime})...", flush=True)
    model = IBPActionClassifierV3(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_expr_layers=args.n_expr_layers,
        n_cross_layers=args.n_cross_layers,
        prime=args.prime
    )
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 10
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0
    history = []

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_metrics']['top1_acc']
        print(f"  Resumed from epoch {checkpoint['epoch']}, best_val_acc={best_val_acc:.4f}", flush=True)

    # Training loop
    print("\nStarting training...", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, args.device)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s):", flush=True)
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"top1={train_metrics['top1_acc']:.4f}, top5={train_metrics['top5_acc']:.4f}", flush=True)
        print(f"  Val:   loss={val_metrics['loss']:.4f}, "
              f"top1={val_metrics['top1_acc']:.4f}, top5={val_metrics['top5_acc']:.4f}", flush=True)

        # Save history
        history_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': scheduler.get_last_lr()[0]
        }
        history.append(history_entry)

        # Save checkpoint if best validation accuracy
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  -> New best val acc! Saved to {checkpoint_path}", flush=True)

        # Test on I[2,0,2,0,1,1,0] on first epoch and every 10 epochs
        if epoch == start_epoch or epoch % 10 == 0:
            print(f"  Testing on I[2,0,2,0,1,1,0]...", flush=True)
            test_expr = {(2, 0, 2, 0, 1, 1, 0): 1}
            success, n_steps, final_expr = ibp_env.reduce_with_model(
                model, test_expr, device=args.device, max_steps=50, verbose=True
            )
            from ibp_env import filter_top_sector, is_master
            non_masters = len([k for k in filter_top_sector(final_expr) if not is_master(k)])
            status = "SUCCESS" if success else f"FAILED ({non_masters} non-masters)"
            print(f"  I[2,0,2,0,1,1,0]: {status} in {n_steps} steps", flush=True)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}", flush=True)

    # Save final model and history
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'args': vars(args)
    }, final_path)

    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("Training complete!", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.4f}", flush=True)
    print(f"Final model saved to: {final_path}", flush=True)
    print(f"Training history saved to: {history_path}", flush=True)


if __name__ == '__main__':
    main()
