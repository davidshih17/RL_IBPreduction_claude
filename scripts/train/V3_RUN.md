# SAILIR v3 — exact training command

The two committed checkpoints
([`checkpoints/sailir_v3_pentagonbox_e29_pre_overfit.pt`](../../checkpoints/sailir_v3_pentagonbox_e29_pre_overfit.pt)
and [`checkpoints/sailir_v3_pentagonbox_e45_best.pt`](../../checkpoints/sailir_v3_pentagonbox_e45_best.pt))
came from a single training run on NERSC Perlmutter, started 2026-06-05 09:08
PDT. This document records the exact command and environment used so the run
can be reproduced. All per-epoch logs are bundled in
[`results/training_log/v3/`](../../results/training_log/v3/).

## Architecture summary

`MODEL_VARIANT=v3` selects [`sailir/classifier_v3.py`](../../sailir/classifier_v3.py)
(`IBPActionClassifierV3`). The defining change vs the published `full` model:

- **State `s`** = (expression terms, target integral, sector mask). The subs
  encoder is removed entirely; the model receives no `sub_*` tensors as direct
  input.
- **Action `a`** = the **transitively-resolved post-substitution equation**
  that the action produces — i.e. the set of (integral, coefficient) terms
  obtained by `apply_resolved_subs(get_raw_equation(ibp_op, seed), resolve_subs(subs))`.
  The `(ibp_op, delta)` handle is not given to the model.
- The post-sub equation is computed on-the-fly in the dataloader's
  `collate_fn` (see `train_classifier.py:make_collate_fn(variant='v3', ...)`);
  no new preprocessing or shard format is required.

## Exact command

Run on a Perlmutter login node from the project root
(`/global/homes/d/dshih/m4539_d/SAILIR_phase2/`):

```bash
tmux new-session -d -s sailir-v3 "
  cd /global/homes/d/dshih/m4539_d/SAILIR_phase2 && \
  TARGET_EPOCHS=100 MAX_ALLOCATIONS=12 \
  MODEL_VARIANT=v3 \
  NUM_WORKERS=8 \
  OUTPUT_DIR=checkpoints/pentagonbox_10x_loop_100_v3 \
  bash scripts/train/perlmutter_train_loop.sh
"
```

The supervisor inside the tmux session repeatedly requests:

```bash
salloc -N 4 -C gpu --gpus 16 -q interactive -t 04:00:00 -A m4539_g \
  bash scripts/train/perlmutter_pentagonbox_10x.sh
```

— 4 nodes × 4 A100 GPUs each = 16 GPUs DDP, 4-hour interactive QOS allocations.
After each allocation hits its walltime (SIGTERM rc=143) the supervisor reads
`last.pt['epoch']`, sleeps 30s, and requests the next allocation. Up to 12
allocations are made (`MAX_ALLOCATIONS=12` → ≤ 48 h compute budget).

`--auto_resume` is on inside `perlmutter_pentagonbox_10x.sh`, so each new
allocation picks up from the latest checkpoint in `OUTPUT_DIR`.

## Effective training arguments

The launcher passes these to `train_classifier.py` (all defaults set inside
`perlmutter_pentagonbox_10x.sh`):

```
--topology         topology_input/pentagonbox
--shards_dir       data/pentagonbox_10x_packed
--n_val_shards     50
--epochs           100
--batch_size       128
--lr               4e-4
--prime            1009
--num_workers      8
--checkpoint_every 1
--log_every        50
--seed             0
--auto_resume
--model_variant    v3
```

- Per-rank batch size 128; effective global batch = 128 × 16 = 2048.
- Optimizer: AdamW, `weight_decay=1e-5` (defaults in `train_classifier.py`).
- All transformer-encoder defaults: `embed_dim=256`, `n_heads=4`,
  `n_expr_layers=2`, `n_cross_layers=2`.

## Results

Run started 2026-06-05 09:08 PDT, supervisor log
[`results/training_log/v3/supervisor.log`](../../results/training_log/v3/supervisor.log).
Concatenated per-iteration training logs in
[`results/training_log/v3/training.log`](../../results/training_log/v3/training.log).

| Checkpoint | Epoch | val_loss | val_top1 | val_top5 | Notes |
|---|---:|---:|---:|---:|---|
| [`sailir_v3_pentagonbox_e29_pre_overfit.pt`](../../checkpoints/sailir_v3_pentagonbox_e29_pre_overfit.pt) | 29 | 0.0404 | 98.54% | 99.95% | last epoch with val/train gap ≈ 0 (Δ=+0.0002 in loss). Use this if you prefer the most-generalising weights. |
| [`sailir_v3_pentagonbox_e45_best.pt`](../../checkpoints/sailir_v3_pentagonbox_e45_best.pt) | 45 | 0.0389 | **98.62%** | 99.95% | lowest val_loss seen so far. By E45 the val/train gap has widened to +0.006 — modest overfitting. |

For reference, the original `full` model trained on the same data plateaued
at val_top1 ≈ **96.08%**.

## Reproducibility notes

- All randomness controlled by `--seed 0`. DDP NCCL is not bit-deterministic
  by default; expect small (< 0.05% top-1) run-to-run variance.
- The `apply_resolved_subs` collate is pure Python and runs in each DataLoader
  worker. With `NUM_WORKERS=8` and `batch_size=128` per rank, the dataloader
  is not the bottleneck on A100s; training is GPU-bound at ~340 ms/batch.
- The supervisor is designed for the interactive QOS (4-node cap, 4-hour
  walltime). On other queues, override `WALLTIME` / `NODES` / `GPUS` env vars
  on the supervisor (see [`PERLMUTTER.md`](PERLMUTTER.md) for the full list).
- If you don't have the m4539_g account, edit the `-A` flag inside
  `perlmutter_train_loop.sh`.
