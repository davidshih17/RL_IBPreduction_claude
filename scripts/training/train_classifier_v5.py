"""
Training script for IBP Action Classifier v5.

Uses packed tensor format with offsets. Includes:
- Target integral as input
- Full substitution encoding (subs_raw)

Usage:
    nohup python -u scripts/training/train_classifier_v5.py \
        --data_dir data/multisector_tensors_v2 > logs/train_v5.log 2>&1 &
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from classifier_v5 import IBPActionClassifierV5


class PackedDatasetV5(Dataset):
    """Dataset using packed format with offsets, including target and subs_raw."""

    def __init__(self, data):
        self.data = data
        self.n_samples = len(data['labels'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        d = self.data

        expr_start, expr_end = d['expr_offsets'][idx].item(), d['expr_offsets'][idx + 1].item()
        sub_start, sub_end = d['sub_offsets'][idx].item(), d['sub_offsets'][idx + 1].item()
        action_start, action_end = d['action_offsets'][idx].item(), d['action_offsets'][idx + 1].item()

        return {
            'expr_integrals': d['expr_integrals'][expr_start:expr_end],
            'expr_coeffs': d['expr_coeffs'][expr_start:expr_end],
            'sub_integrals': d['sub_integrals'][sub_start:sub_end],
            'subs_raw': d['subs_raw'][idx],  # Full substitution structure
            'action_ibp_ops': d['action_ibp_ops'][action_start:action_end],
            'action_deltas': d['action_deltas'][action_start:action_end],
            'sector_mask': d['sector_masks'][idx],
            'target_integral': d['target_integrals'][idx],  # Target integral
            'label': d['labels'][idx],
        }


MAX_REPLACEMENT_TERMS = 20  # Max replacement terms per substitution

def collate_fn(samples):
    """Pad variable-length tensors to batch max and stack."""
    batch_size = len(samples)

    max_expr = max(len(s['expr_integrals']) for s in samples)
    max_sub = max(len(s['subs_raw']) for s in samples) if any(len(s['subs_raw']) > 0 for s in samples) else 1
    max_action = max(len(s['action_ibp_ops']) for s in samples)

    expr_integrals = torch.zeros(batch_size, max_expr, 7, dtype=torch.long)
    expr_coeffs = torch.zeros(batch_size, max_expr, dtype=torch.long)
    expr_mask = torch.zeros(batch_size, max_expr, dtype=torch.bool)

    # Batched substitution tensors
    sub_keys = torch.zeros(batch_size, max_sub, 7, dtype=torch.long)
    sub_repl_ints = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, 7, dtype=torch.long)
    sub_repl_coeffs = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.long)
    sub_repl_mask = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.bool)
    sub_mask = torch.zeros(batch_size, max_sub, dtype=torch.bool)

    action_ibp_ops = torch.zeros(batch_size, max_action, dtype=torch.long)
    action_deltas = torch.zeros(batch_size, max_action, 7, dtype=torch.long)
    action_mask = torch.zeros(batch_size, max_action, dtype=torch.bool)

    sector_masks = torch.zeros(batch_size, 6, dtype=torch.long)
    target_integrals = torch.zeros(batch_size, 7, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, s in enumerate(samples):
        n_expr = len(s['expr_integrals'])
        n_action = len(s['action_ibp_ops'])

        expr_integrals[i, :n_expr] = s['expr_integrals'].long()
        expr_coeffs[i, :n_expr] = s['expr_coeffs'].long()
        expr_mask[i, :n_expr] = True

        # Process subs_raw into batched tensors
        subs = s['subs_raw']
        for j, sub in enumerate(subs[:max_sub]):
            sub_mask[i, j] = True
            sub_keys[i, j] = torch.tensor(sub[0], dtype=torch.long)
            replacement = sub[1]
            n_repl = min(len(replacement), MAX_REPLACEMENT_TERMS)
            for r in range(n_repl):
                sub_repl_ints[i, j, r] = torch.tensor(replacement[r][0], dtype=torch.long)
                sub_repl_coeffs[i, j, r] = replacement[r][1]
                sub_repl_mask[i, j, r] = True

        action_ibp_ops[i, :n_action] = s['action_ibp_ops'].long()
        action_deltas[i, :n_action] = s['action_deltas'].long()
        action_mask[i, :n_action] = True

        sector_masks[i] = s['sector_mask'].long()
        target_integrals[i] = s['target_integral'].long()
        labels[i] = s['label'].long()

    return {
        'expr_integrals': expr_integrals,
        'expr_coeffs': expr_coeffs,
        'expr_mask': expr_mask,
        'sub_keys': sub_keys,
        'sub_repl_ints': sub_repl_ints,
        'sub_repl_coeffs': sub_repl_coeffs,
        'sub_repl_mask': sub_repl_mask,
        'sub_mask': sub_mask,
        'action_ibp_ops': action_ibp_ops,
        'action_deltas': action_deltas,
        'action_mask': action_mask,
        'sector_mask': sector_masks,
        'target_integral': target_integrals,
        'labels': labels,
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral']
        )

        loss = F.cross_entropy(logits, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = batch['labels'].size(0)
        total_loss += loss.item() * bs
        total_top1 += (logits.argmax(-1) == batch['labels']).sum().item()
        _, top5 = logits.topk(min(5, logits.size(1)), dim=-1)
        total_top5 += (top5 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        total_samples += bs

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} batch {batch_idx+1}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, top1={total_top1/total_samples:.3f}", flush=True)

    return {'loss': total_loss/total_samples, 'top1_acc': total_top1/total_samples, 'top5_acc': total_top5/total_samples}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral']
        )
        loss = F.cross_entropy(logits, batch['labels'])
        bs = batch['labels'].size(0)
        total_loss += loss.item() * bs
        total_top1 += (logits.argmax(-1) == batch['labels']).sum().item()
        _, top5 = logits.topk(min(5, logits.size(1)), dim=-1)
        total_top5 += (top5 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        total_samples += bs

    return {'loss': total_loss/total_samples, 'top1_acc': total_top1/total_samples, 'top5_acc': total_top5/total_samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with train.pt, val.pt')
    parser.add_argument('--output_dir', type=str, default='checkpoints/classifier_v5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_expr_layers', type=int, default=2)
    parser.add_argument('--n_cross_layers', type=int, default=2)
    parser.add_argument('--n_subs_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("IBP Action Classifier v5 Training", flush=True)
    print("  - Target as input", flush=True)
    print("  - Full substitution encoding", flush=True)
    print("=" * 70, flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("\nLoading packed tensors...", flush=True)
    t0 = time.time()
    train_data = torch.load(data_dir / 'train.pt', weights_only=False)  # weights_only=False for subs_raw
    val_data = torch.load(data_dir / 'val.pt', weights_only=False)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    # Check that new fields exist
    assert 'target_integrals' in train_data, "Missing target_integrals! Did you repack with preprocess_to_tensors.py?"
    assert 'subs_raw' in train_data, "Missing subs_raw! Did you repack with preprocess_to_tensors.py?"
    print(f"  target_integrals shape: {train_data['target_integrals'].shape}", flush=True)
    print(f"  subs_raw count: {len(train_data['subs_raw'])}", flush=True)

    train_dataset = PackedDatasetV5(train_data)
    val_dataset = PackedDatasetV5(val_data)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ibp_env import set_prime
    set_prime(args.prime)

    model = IBPActionClassifierV5(
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_expr_layers=args.n_expr_layers, n_cross_layers=args.n_cross_layers,
        n_subs_layers=args.n_subs_layers, prime=args.prime
    )
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)

    start_epoch, best_val_loss = 1, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_metrics']['loss']
        print(f"Resumed from epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f}", flush=True)

    print("\nStarting training...", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_m = evaluate(model, val_loader, args.device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.1f}s):", flush=True)
        print(f"  Train: loss={train_m['loss']:.4f}, top1={train_m['top1_acc']:.4f}, top5={train_m['top5_acc']:.4f}", flush=True)
        print(f"  Val:   loss={val_m['loss']:.4f}, top1={val_m['top1_acc']:.4f}, top5={val_m['top5_acc']:.4f}", flush=True)

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'val_metrics': val_m,
                        'args': vars(args)}, output_dir / 'best_model.pt')
            print(f"  -> New best val loss! Saved.", flush=True)

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'val_metrics': val_m,
                        'args': vars(args)}, output_dir / f'checkpoint_epoch{epoch}.pt')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_metrics': val_m,
                'args': vars(args)}, output_dir / 'final_model.pt')

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}", flush=True)


if __name__ == '__main__':
    main()
