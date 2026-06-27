"""
Timing diagnostic — isolate where pentagonbox forward time goes.
num_workers=0 (everything in main process, debuggable), small batches.
"""
import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from train_classifier import PackedDatasetV5, make_collate_fn  # noqa: E402
from sharded_dataset import ShardedIBPDataset, discover_shards  # noqa: E402
from sailir.topology import Topology  # noqa: E402
from sailir import ibp_env  # noqa: E402
from sailir.classifier import IBPActionClassifier  # noqa: E402


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def t(label, t0):
    cuda_sync()
    dt = time.time() - t0
    print(f"  [{label}] {dt*1000:.1f} ms", flush=True)
    return time.time()


def main():
    print("=== Setup ===", flush=True)
    topology = Topology.from_dir(str(ROOT / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    ibp_env.set_prime(1009)
    print(f"n_indices={topology.n_indices}, n_actions={topology.n_actions}", flush=True)

    print(f"CUDA visible: {torch.cuda.device_count()} devices", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)

    shards_dir = ROOT / 'data' / 'pentagonbox_10x_packed'
    train_shards = discover_shards(shards_dir, 'train')[:1]
    print(f"Using 1 shard: {train_shards[0]}", flush=True)

    t0 = time.time()
    ds = ShardedIBPDataset(train_shards, shuffle=True, buffer_shards=1, seed=0)
    ds.set_epoch(0)
    t0 = t("dataset construct", t0)

    collate = make_collate_fn(n_indices=topology.n_indices, n_denominators=topology.n_denominators)

    # ---- Test 1: dataset iteration only (no model) ----
    print("\n=== Test 1: dataset+collate, 5 batches, batch_size=4 ===", flush=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate,
                        num_workers=0, pin_memory=False)
    t0 = time.time()
    for i, batch in enumerate(loader):
        t0 = t(f"batch {i} (bs=4)", t0)
        if i >= 4: break

    # ---- Test 2: dataset iteration only, batch_size=64 ----
    print("\n=== Test 2: dataset+collate, 3 batches, batch_size=64 ===", flush=True)
    ds.set_epoch(1)  # fresh
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate,
                        num_workers=0, pin_memory=False)
    t0 = time.time()
    for i, batch in enumerate(loader):
        t0 = t(f"batch {i} (bs=64)", t0)
        if i >= 2: break

    # Print one batch's shapes so we know what the model sees
    print("\nBatch shapes:", flush=True)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}", flush=True)

    # ---- Test 3: model forward on 1 GPU, bs=4 ----
    print("\n=== Test 3: model forward, 1 GPU, bs=4 ===", flush=True)
    model = IBPActionClassifier(
        embed_dim=256, n_heads=4,
        n_expr_layers=2, n_cross_layers=2, n_subs_layers=2, prime=1009,
        n_indices=topology.n_indices, n_denominators=topology.n_denominators, n_ibp_ops=topology.n_actions,
    ).to('cuda:0')
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)

    ds.set_epoch(2)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate,
                        num_workers=0, pin_memory=False)
    for i, batch in enumerate(loader):
        batch = {k: v.to('cuda:0', non_blocking=False) for k, v in batch.items()}
        cuda_sync()
        t0 = time.time()
        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral']
        )
        t0 = t(f"fwd batch {i}", t0)
        loss = F.cross_entropy(logits, batch['labels'])
        loss.backward()
        t0 = t(f"bwd batch {i}", t0)
        opt.step(); opt.zero_grad()
        t0 = t(f"opt batch {i}", t0)
        if i >= 2: break

    # ---- Test 4: model forward on 1 GPU, bs=64 ----
    print("\n=== Test 4: model forward, 1 GPU, bs=64 ===", flush=True)
    ds.set_epoch(3)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate,
                        num_workers=0, pin_memory=False)
    for i, batch in enumerate(loader):
        batch = {k: v.to('cuda:0', non_blocking=False) for k, v in batch.items()}
        cuda_sync()
        t0 = time.time()
        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral']
        )
        t0 = t(f"fwd batch {i}", t0)
        loss = F.cross_entropy(logits, batch['labels'])
        loss.backward()
        t0 = t(f"bwd batch {i}", t0)
        opt.step(); opt.zero_grad()
        t0 = t(f"opt batch {i}", t0)
        if i >= 2: break

    # ---- Test 5: DataParallel, bs=384 (target run config) ----
    print("\n=== Test 5: DataParallel 3 GPU, bs=384 ===", flush=True)
    if torch.cuda.device_count() >= 3:
        dp_model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        ds.set_epoch(4)
        loader = DataLoader(ds, batch_size=384, shuffle=False, collate_fn=collate,
                            num_workers=0, pin_memory=False)
        for i, batch in enumerate(loader):
            batch = {k: v.to('cuda:0', non_blocking=False) for k, v in batch.items()}
            cuda_sync()
            t0 = time.time()
            logits, _ = dp_model(
                batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
                batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
                batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
                batch['sector_mask'], batch['target_integral']
            )
            t0 = t(f"DP fwd batch {i}", t0)
            loss = F.cross_entropy(logits, batch['labels'])
            loss.backward()
            t0 = t(f"DP bwd batch {i}", t0)
            opt.step(); opt.zero_grad()
            t0 = t(f"DP opt batch {i}", t0)
            if i >= 1: break

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
