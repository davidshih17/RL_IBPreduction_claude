"""
Training script for IBP Action Classifier v5.

Uses packed tensor format with offsets. Includes:
- Target integral as input
- Full substitution encoding (subs_raw)

Usage:
    nohup python -u scripts/train/train_classifier.py \
        --data_dir data/multisector_tensors_v2 > logs/train_v5.log 2>&1 &
"""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'sailir'))
from classifier import IBPActionClassifier


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


def make_collate_fn(n_indices, n_denominators):
    """Build a topology-aware collate_fn closure."""
    def collate_fn(samples):
        return _collate(samples, n_indices, n_denominators)
    return collate_fn


def _collate(samples, n_indices, n_denominators):
    """Pad variable-length tensors to batch max and stack."""
    batch_size = len(samples)

    max_expr = max(len(s['expr_integrals']) for s in samples)
    max_sub = max(len(s['subs_raw']) for s in samples) if any(len(s['subs_raw']) > 0 for s in samples) else 1
    max_action = max(len(s['action_ibp_ops']) for s in samples)

    expr_integrals = torch.zeros(batch_size, max_expr, n_indices, dtype=torch.long)
    expr_coeffs = torch.zeros(batch_size, max_expr, dtype=torch.long)
    expr_mask = torch.zeros(batch_size, max_expr, dtype=torch.bool)

    # Batched substitution tensors
    sub_keys = torch.zeros(batch_size, max_sub, n_indices, dtype=torch.long)
    sub_repl_ints = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, n_indices, dtype=torch.long)
    sub_repl_coeffs = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.long)
    sub_repl_mask = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.bool)
    sub_mask = torch.zeros(batch_size, max_sub, dtype=torch.bool)

    action_ibp_ops = torch.zeros(batch_size, max_action, dtype=torch.long)
    action_deltas = torch.zeros(batch_size, max_action, n_indices, dtype=torch.long)
    action_mask = torch.zeros(batch_size, max_action, dtype=torch.bool)

    sector_masks = torch.zeros(batch_size, n_denominators, dtype=torch.long)
    target_integrals = torch.zeros(batch_size, n_indices, dtype=torch.long)
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


def train_epoch(model, dataloader, optimizer, device, epoch, log_every=1, total_batches=None,
                ddp_enabled=False):
    model.train()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    denom = f"/{total_batches}" if total_batches else ""
    t_prev = time.time()

    # Join handles uneven per-rank iteration counts: when one rank's iterator
    # is exhausted, Join provides "shadow" all-reduces (zero gradients) on its
    # behalf so the still-active ranks' backward all-reduces stay aligned with
    # NCCL's strict ordering requirement. Without this, uneven sample counts
    # across ranks cause NCCL op queues to drift out of order at the
    # train→val boundary, which manifests as a 10-min watchdog timeout.
    join_cm = Join([model]) if ddp_enabled else contextlib.nullcontext()

    with join_cm:
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

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                now = time.time()
                dt_ms = (now - t_prev) * 1000 / log_every
                t_prev = now
                print(f"  E{epoch} b{batch_idx+1}{denom} bs={bs} loss={loss.item():.4f} "
                      f"top1={total_top1/total_samples:.3f} {dt_ms:.0f}ms/batch", flush=True)

    if total_samples == 0:
        # Edge case: rank's shard list was empty (e.g., smoke test with fewer
        # shards than ranks). Return safe defaults so reduce_metrics still works.
        return {'loss': 0.0, 'top1_acc': 0.0, 'top5_acc': 0.0}
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

    if total_samples == 0:
        return {'loss': 0.0, 'top1_acc': 0.0, 'top5_acc': 0.0}
    return {'loss': total_loss/total_samples, 'top1_acc': total_top1/total_samples, 'top5_acc': total_top5/total_samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory '
                             '(determines n_indices, n_denominators, n_actions)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Single-pack mode: directory with train.pt/val.pt. '
                             'Mutually exclusive with --shards_dir.')
    parser.add_argument('--shards_dir', type=str, default=None,
                        help='Sharded mode: directory containing shard_*/train.pt etc. '
                             'Uses ShardedIBPDataset (memory-bounded streaming).')
    parser.add_argument('--n_val_shards', type=str, default='50',
                        help='How many shards to use for validation each epoch '
                             '(int, or "all"). Ignored in --data_dir mode.')
    parser.add_argument('--max_train_shards', type=int, default=0,
                        help='Cap the number of train shards (0 = use all). For smoke tests.')
    parser.add_argument('--buffer_shards', type=int, default=4,
                        help='Number of shards held in the train shuffle buffer '
                             '(per DataLoader worker). Larger = better mixing, more RAM.')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional path to write a per-epoch metrics TSV (epoch, train_loss, ...).')
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume from.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='If <output_dir>/last.pt exists, resume from it. '
                             'Overrides --resume when both are set and last.pt exists.')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Also save a numbered checkpoint every N epochs (in addition to last.pt).')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Print per-batch metrics every N batches (1 = every batch).')
    args = parser.parse_args()

    if (args.data_dir is None) == (args.shards_dir is None):
        parser.error('Exactly one of --data_dir or --shards_dir is required.')

    # DDP detection: torchrun sets LOCAL_RANK/RANK/WORLD_SIZE env vars.
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    ddp_enabled = local_rank >= 0
    if ddp_enabled:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        rank = 0
        world_size = 1
        device = args.device
    is_main = (rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # Load topology to determine model dimensions.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sailir.topology import Topology
    topology = Topology.from_dir(args.topology)
    n_indices = topology.n_indices
    n_denominators = topology.n_denominators
    n_actions = topology.n_actions

    log("=" * 70)
    log("IBP Action Classifier v5 Training")
    log("  - Target as input")
    log("  - Full substitution encoding")
    if ddp_enabled:
        log(f"  - DDP: rank {rank}/{world_size} on cuda:{local_rank}")
    log("=" * 70)
    for k, v in vars(args).items():
        log(f"  {k}: {v}")

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None

    collate = make_collate_fn(n_indices=n_indices, n_denominators=n_denominators)

    if args.shards_dir is not None:
        # Sharded streaming mode — memory-bounded.
        from sharded_dataset import (
            ShardedIBPDataset, discover_shards, estimate_samples_per_epoch,
        )

        log(f"\nDiscovering shards in {args.shards_dir}...")
        train_shards = discover_shards(args.shards_dir, 'train')
        val_shards_all = discover_shards(args.shards_dir, 'val')
        log(f"  train shards: {len(train_shards)}")
        log(f"  val shards available: {len(val_shards_all)}")
        if not train_shards:
            raise RuntimeError(f"No train.pt shards found under {args.shards_dir}")

        if args.max_train_shards > 0:
            train_shards = train_shards[: args.max_train_shards]
            log(f"  --max_train_shards: capped train to {len(train_shards)} shards")

        if args.n_val_shards == 'all':
            val_shards = val_shards_all
        else:
            val_shards = val_shards_all[: int(args.n_val_shards)]
        log(f"  val shards selected: {len(val_shards)}")

        # In DDP mode, truncate to a multiple of world_size so every rank
        # iterates the same number of shards → matching batch counts at
        # all-reduce time (DDP would otherwise hang on the trailing batches).
        if ddp_enabled and world_size > 1:
            for name, shards_list in (('train', train_shards), ('val', val_shards)):
                n_keep = (len(shards_list) // world_size) * world_size
                if n_keep < len(shards_list):
                    log(f"  {name}: truncating to {n_keep} shards (world_size={world_size}; "
                        f"dropping {len(shards_list)-n_keep})")
                shards_list[:] = shards_list[:n_keep]

        # Manifest gives exact samples/epoch; build once and cache.
        manifest_path = Path(args.shards_dir) / 'manifest.json'
        if manifest_path.is_file():
            with open(manifest_path) as f:
                manifest = json.load(f)
            log(f"  manifest: {manifest_path} (cached)")
        else:
            log(f"  manifest not found — estimating sample counts from 5 shards")
            manifest = None

        n_train_samples = estimate_samples_per_epoch(train_shards, manifest, 'train')
        n_val_samples = estimate_samples_per_epoch(val_shards, manifest, 'val')
        log(f"  ~train samples/epoch (global): {n_train_samples:,}")
        log(f"  ~val   samples/epoch (global): {n_val_samples:,}")

        train_dataset = ShardedIBPDataset(
            train_shards, shuffle=True, buffer_shards=args.buffer_shards, seed=args.seed,
            rank=rank, world_size=world_size,
        )
        val_dataset = ShardedIBPDataset(
            val_shards, shuffle=False, buffer_shards=1, seed=args.seed,
            rank=rank, world_size=world_size,
        )
        # IterableDataset → shuffle=False at DataLoader; shuffling is handled
        # inside the dataset. persistent_workers=False so each iter(loader)
        # re-pickles the dataset to fresh workers, carrying the updated
        # _epoch from set_epoch() — persistent workers would freeze at _epoch=0.
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False,
        )
        # Per-rank batch count (used only for the progress denominator string).
        approx_train_batches = max(1, n_train_samples // (args.batch_size * world_size))
        approx_val_batches = max(1, n_val_samples // (args.batch_size * world_size))
        log(f"  ~train batches/epoch/rank: {approx_train_batches:,}")
        log(f"  ~val   batches/epoch/rank: {approx_val_batches:,}")
    else:
        log("\nLoading packed tensors...")
        t0 = time.time()
        train_data = torch.load(data_dir / 'train.pt', weights_only=False)
        val_data = torch.load(data_dir / 'val.pt', weights_only=False)
        log(f"Loaded in {time.time() - t0:.1f}s")

        assert 'target_integrals' in train_data, "Missing target_integrals!"
        assert 'subs_raw' in train_data, "Missing subs_raw!"
        log(f"  target_integrals shape: {train_data['target_integrals'].shape}")
        log(f"  subs_raw count: {len(train_data['subs_raw'])}")

        train_dataset = PackedDatasetV5(train_data)
        val_dataset = PackedDatasetV5(val_data)
        log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
        approx_train_batches = len(train_loader)
        approx_val_batches = len(val_loader)

        log(f"Train batches: {approx_train_batches}, Val batches: {approx_val_batches}")

    from sailir import ibp_env
    ibp_env.init_from_topology(topology)
    ibp_env.set_prime(args.prime)

    model = IBPActionClassifier(
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_expr_layers=args.n_expr_layers, n_cross_layers=args.n_cross_layers,
        n_subs_layers=args.n_subs_layers, prime=args.prime,
        n_indices=n_indices, n_denominators=n_denominators, n_ibp_ops=n_actions,
    )
    model = model.to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if ddp_enabled:
        # DDP: one process per GPU, model lives permanently on each. Only
        # gradients cross the interconnect, via NCCL all-reduce overlapped
        # with backward.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        log(f"Using DistributedDataParallel: world_size={world_size}")
        base_model = model.module
    elif device == 'cuda' and torch.cuda.device_count() > 1:
        # Legacy DataParallel path — known to be slow on SYS-grade interconnects
        # (separate PCIe roots). Kept for environments where it works (NVLink
        # or shared PCIe switch); torchrun → DDP is the preferred path here.
        n_gpu = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        log(f"Using DataParallel over {n_gpu} GPUs: {names}")
        model = torch.nn.DataParallel(model)
        base_model = model.module
    else:
        base_model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)

    def _full_state(epoch, val_metrics, train_metrics=None):
        # Persist enough state that a resumed run continues the cosine LR
        # schedule and best-val tracking without drift.
        return {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }

    start_epoch, best_val_loss = 1, float('inf')
    resume_path = None
    if args.auto_resume and (output_dir / 'last.pt').is_file():
        resume_path = output_dir / 'last.pt'
        log(f"--auto_resume: found {resume_path}")
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path is not None:
        # map_location keeps each rank's parameters on its own device.
        ckpt = torch.load(resume_path, weights_only=False, map_location=device)
        base_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', ckpt['val_metrics']['loss'])
        log(f"Resumed from epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f}")

    # Per-epoch metrics log — only rank 0 writes.
    log_fp = None
    if args.log_file and is_main:
        log_fp = open(args.log_file, 'a')
        if log_fp.tell() == 0:
            log_fp.write("epoch\ttrain_loss\ttrain_top1\ttrain_top5\tval_loss\tval_top1\tval_top5\tlr\twall_s\n")
            log_fp.flush()

    def reduce_metrics(m):
        """All-reduce a dict of scalar metrics across DDP ranks (mean)."""
        if not ddp_enabled:
            return m
        out = {}
        for k, v in m.items():
            t = torch.tensor(float(v), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = (t.item() / world_size)
        return out

    log("\nStarting training...")
    log("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        train_m_local = train_epoch(
            model, train_loader, optimizer, device, epoch,
            log_every=args.log_every if is_main else 0,
            total_batches=approx_train_batches,
            ddp_enabled=ddp_enabled,
        )
        val_m_local = evaluate(model, val_loader, device)
        scheduler.step()

        # Aggregate across ranks so all log/save decisions use the global view.
        train_m = reduce_metrics(train_m_local)
        val_m = reduce_metrics(val_m_local)

        wall = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch}/{args.epochs} ({wall:.1f}s, lr={cur_lr:.2e}):")
        log(f"  Train: loss={train_m['loss']:.4f}, top1={train_m['top1_acc']:.4f}, top5={train_m['top5_acc']:.4f}")
        log(f"  Val:   loss={val_m['loss']:.4f}, top1={val_m['top1_acc']:.4f}, top5={val_m['top5_acc']:.4f}")

        is_best = val_m['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_m['loss']

        if is_main:
            # Always save last.pt — atomic via temp-then-rename.
            state = _full_state(epoch, val_m, train_m)
            tmp = output_dir / 'last.pt.tmp'
            torch.save(state, tmp)
            os.replace(tmp, output_dir / 'last.pt')

            if is_best:
                torch.save(state, output_dir / 'best_model.pt')
                log(f"  -> New best val loss! Saved best_model.pt")

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                torch.save(state, output_dir / f'checkpoint_epoch{epoch}.pt')

            if log_fp is not None:
                log_fp.write(f"{epoch}\t{train_m['loss']:.6f}\t{train_m['top1_acc']:.6f}\t{train_m['top5_acc']:.6f}"
                             f"\t{val_m['loss']:.6f}\t{val_m['top1_acc']:.6f}\t{val_m['top5_acc']:.6f}"
                             f"\t{cur_lr:.6e}\t{wall:.1f}\n")
                log_fp.flush()

        if ddp_enabled:
            # Make sure non-main ranks wait for rank-0 to finish writing before
            # the next epoch starts (purely defensive — no shared file system race
            # is possible here, but the barrier keeps the per-epoch logs ordered).
            dist.barrier()

    if is_main:
        torch.save(_full_state(args.epochs, val_m, train_m), output_dir / 'final_model.pt')
        if log_fp is not None:
            log_fp.close()
        log(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
