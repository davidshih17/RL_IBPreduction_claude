# SAILIR — Training phase

This directory is the **training phase** of SAILIR: it trains the action
classifier on the packed-tensor shards produced by `../data-gen/`, for **any
integral topology**. The result is the model that `../reduction/` then uses to
drive the beam-search reduction. The shared core library is the `../sailir/`
package.

---

## 1. What the script does

`train_classifier.py` trains a poly-encoder cross-attention model that, at each
reduction step, scores the set of legal IBP/LI actions and predicts the one
that reduces toward masters. The supervised signal — `(state → correct-action)`
— comes directly from the packed shards under `data/<dataset>_packed/shard_*/`.

Two model variants ship:

| Variant | Module | What it does | When to use |
|---------|--------|--------------|-------------|
| `full` | [`../sailir/classifier.py`](../sailir/classifier.py) | `IBPActionClassifier` — encodes expression, target, sector, **and the substitution history** via a `FullSubstitutionEncoder`. | Use to reproduce the committed reference checkpoint `../checkpoints/pentagonbox_10x_loop_100/best_model.pt`. |
| `nosubs` (**recommended**) | [`../sailir/classifier_nosubs.py`](../sailir/classifier_nosubs.py) | `IBPActionClassifierNoSubs` — same architecture as `full` with the subs encoder **removed**; `state_combine` shrunk from 4·embed_dim → 3·embed_dim. | Recommended for new training runs (see §4). |

Pick with `--model_variant {full,nosubs}`. Checkpoints from the two variants
are NOT interchangeable; use a distinct `--output_dir` per variant.

Both variants are **topology-agnostic** — they read `n_indices`,
`n_denominators`, `n_actions` from the `Topology` object at construction
time. Point them at a different `topology_input/<family>/` and they train
that family's data.

---

## 2. Prerequisites

- **Python** (with PyTorch ≥ 2.0). The training step needs only `torch` +
  `numpy`; no Cython kernels are required at train time (those are needed
  for `../reduction/` and for parts of `../data-gen/`).
- **Packed shards** produced by `../data-gen/`:
  `data/<dataset>_packed/shard_*/{train,val,test}.pt`. See
  `../data-gen/README.md` for the format and recipe.
- **A topology directory** at `topology_input/<family>/` (the same one used by
  data-gen). The script reads `n_indices`, `n_denominators`, `n_actions` from
  it via `sailir.topology.Topology.from_dir`.
- **GPU(s)**. The script auto-detects DDP via the `LOCAL_RANK`/`RANK`/
  `WORLD_SIZE` env vars (set by `torchrun` or `srun --gpus-per-task=1`).
  On a single GPU it just runs single-process.

---

## 3. The data format

Each packed shard contains the per-step supervised samples emitted by
`../data-gen/generate_multisector_data.py`. Per-sample fields (loaded by
`sharded_dataset.py` and assembled into batches by the collate function):

| field | shape (per sample) | meaning |
|-------|--------------------|---------|
| `expr_integrals` | `(L_expr, n_indices)` | the variable-length expression at this step |
| `expr_coeffs`    | `(L_expr,)`           | finite-field coefficients of those terms |
| `target_integral`| `(n_indices,)`        | the **highest-weight** non-master integral in `expr` (by `(-r, -s, +lex)`) — the one being reduced this step |
| `sector_mask`    | `(n_denominators,)`   | 0/1 mask of the current sector |
| `subs_raw`       | nested list           | substitution history (used only by `full`; ignored by `nosubs`) |
| `action_ibp_ops`, `action_deltas` | `(A,)` / `(A, n_indices)` | the variable-length set of legal IBP/LI actions at this state |
| `label`          | scalar                | index into the action set — the **correct action** (the one the data-gen recorded as moving toward masters) |

The dataset class `ShardedIBPDataset` ([`sharded_dataset.py`](sharded_dataset.py))
streams these per-rank with a shuffle buffer of `--buffer_shards` shards in RAM
(memory-bounded; never holds the whole dataset).

---

## 4. Why `nosubs` is recommended

The `full` model's `FullSubstitutionEncoder` is ~38.5% of total params (~3.0M
of 7.7M), but ablation (zero / shuffle / random sub-content perturbations)
showed the trained `full` model uses it as essentially a **1-bit "has-subs"
detector** — random subs produce *bit-identical* predictions to true subs at
every sub-count bucket. The full subs path is therefore wasted capacity for
this task. `nosubs` drops it entirely:

| | `full` | `nosubs` | delta |
|---|---:|---:|---:|
| Parameters | 8.12 M | 4.90 M | **−39.6%** |
| Peak training mem (B=128) | 13,269 MB | 12,515 MB | −5.7% |
| Training step (ms) | 927 | 813 | **−12.3%** |
| Best val top-1 (pentagonbox_10x, 100-epoch run) | 96.08% (E76) | 96.08% (E62) | **±0.00 pp** |

(The committed reference checkpoint `checkpoints/pentagonbox_10x_loop_100/best_model.pt`
was trained with `full` before this finding was made.)

---

## 5. Quick start — single machine, multi-GPU

The minimum invocation, using `torchrun` to bring up DDP across all GPUs on
one machine:

```bash
SAILIR_DIR=/path/to/SAILIR                          # this repo's root
cd $SAILIR_DIR

torchrun --standalone --nproc_per_node=gpu \
    training/train_classifier.py \
        --topology         topology_input/pentagonbox \
        --shards_dir       data/pentagonbox_10x_packed \
        --output_dir       checkpoints/my_nosubs_run \
        --model_variant    nosubs \
        --batch_size       128 \
        --lr               4e-4 \
        --epochs           100 \
        --num_workers      8 \
        --prime            1009 \
        --checkpoint_every 1 \
        --log_every        50 \
        --auto_resume \
        --n_val_shards     50
```

Notes:

- `--nproc_per_node=gpu` launches one rank per visible GPU. For a single GPU
  just drop `torchrun` and run `python -u training/train_classifier.py ...`.
- `--lr 4e-4` is the per-rank-batch-128 setting that produced the reference
  checkpoint. If you change `--batch_size`, scale `--lr` accordingly
  (linear or sqrt rule).
- `--auto_resume` makes the run pick up from `<output_dir>/last.pt` if it
  exists, so re-launching the same command after a crash resumes cleanly.
- `--prime 1009` must match the prime used at data-gen time (see
  `../data-gen/README.md` §7).

For a smoke test, add `--max_train_shards 64 --n_val_shards 16 --epochs 5`.

---

## 6. Quick start — NERSC Perlmutter (interactive queue)

The `nersc_perlmutter/` subdirectory ships the **unattended multi-allocation
supervisor** used to train `pentagonbox_10x` on Perlmutter's 4-node × 4-hour
interactive GPU queue. See [`nersc_perlmutter/README.md`](nersc_perlmutter/README.md)
for the full procedure. Short version:

```bash
ssh perlmutter
tmux new -s sailir
cd /path/to/SAILIR
TARGET_EPOCHS=100 MAX_ALLOCATIONS=12 \
    MODEL_VARIANT=nosubs \
    OUTPUT_DIR=checkpoints/pentagonbox_10x_loop_100_nosubs \
    NUM_WORKERS=8 \
    bash training/nersc_perlmutter/train_loop.sh
# Ctrl-B D to detach; `tmux a -t sailir` to re-attach later.
```

The supervisor requests one 4-hour interactive allocation at a time, runs
`allocation.sh` inside it (which `srun`s 16 ranks), waits for `salloc` to hit
its walltime (SIGTERM), reads `last.pt['epoch']`, and re-`salloc`s. Stops when
the latest checkpoint reaches `TARGET_EPOCHS` or after `MAX_ALLOCATIONS` tries.

You don't need NERSC — the same `train_classifier.py` runs on any DDP-capable
cluster or single multi-GPU box (§5).

---

## 7. Files in this directory

| File | Role |
|------|------|
| [`train_classifier.py`](train_classifier.py) | **core** — DDP-aware training loop; `--model_variant {full,nosubs}` |
| [`sharded_dataset.py`](sharded_dataset.py) | streaming shard loader with rank-aware partitioning + shuffle buffer; builds/loads `manifest.json` for exact-sample-count alignment |
| [`inspect_shard.py`](inspect_shard.py) | sanity utility — load one packed shard and print field shapes / per-field dtypes / a few sample rows |
| [`nersc_perlmutter/`](nersc_perlmutter/) | optional: unattended-supervisor + per-allocation launcher for NERSC's 4-hour interactive QOS |
| [`archive/`](archive/) | superseded Condor-era training scripts (per-sector finetune jobs, etc.) — kept for historical reference; not part of the supported path |

---

## 8. Parameters & scaling

Defaults match the production `pentagonbox_10x` run; tune as needed.

| Flag | Default | Notes |
|------|--------:|-------|
| `--batch_size` | 256 | Per-rank batch. Effective batch = `batch_size × world_size`. Saved envelope on A100-40GB: **128** safe, **192** OK, **256** OOM (for `full`/`nosubs`). |
| `--lr`         | 4e-4 | At `batch_size=128 × world_size=16` (effective 2048). Scale linearly with effective batch. |
| `--epochs`     | 30   | Production `pentagonbox_10x` used 100. Val top-1 plateaus around E60–E76. |
| `--embed_dim`  | 256  | Embedding width across all encoders. |
| `--n_heads`    | 4    | Cross-attention heads. |
| `--n_expr_layers` | 2 | Expression-encoder transformer layers. |
| `--n_cross_layers`| 2 | Action-scorer cross-attention layers. |
| `--n_subs_layers` | 2 | Substitution-encoder layers (used only by `full`; ignored by `nosubs`). |
| `--prime`      | 1009 | Finite field `GF(p)`; **must match data-gen**. |
| `--buffer_shards` | 4 | Shards held in RAM per DataLoader worker for shuffling. Larger = better mixing, more RAM. |
| `--num_workers` | 4   | DataLoader workers per rank. |
| `--n_val_shards`| 50  | Validation shard count per epoch. Set `all` to use every shard in the val split. |
| `--max_train_shards` | 0 | Cap train shards (0 = all). For smoke tests. |
| `--checkpoint_every` | 5 | Save numbered checkpoint every N epochs (in addition to `last.pt` and `best_model.pt`, both saved every epoch). |
| `--auto_resume` | off | Resume from `<output_dir>/last.pt` if present. |

DDP throughput on Perlmutter (4 nodes × 4 A100, NVLink within node): ~340 ms /
batch at `--batch_size 128`, ~26 min / epoch on `pentagonbox_10x_packed`
(~10.8M samples / epoch global, ~675K / rank).

---

## 9. Sanity checks before a full run

1. **Inspect one shard** to confirm the field set + dtypes:
   ```bash
   python training/inspect_shard.py data/<dataset>_packed/shard_0/train.pt
   ```
   The `target_integral` index length must equal `topology.n_indices`; the
   `sector_mask` length must equal `topology.n_denominators`.

2. **Smoke training** — 5 epochs on 64 train shards (use `world_size` of GPUs):
   ```bash
   torchrun --standalone --nproc_per_node=gpu \
       training/train_classifier.py \
           --topology topology_input/pentagonbox \
           --shards_dir data/pentagonbox_10x_packed \
           --output_dir checkpoints/smoke_$(date +%Y%m%d) \
           --model_variant nosubs \
           --max_train_shards 64 --n_val_shards 16 --epochs 5 \
           --batch_size 128 --lr 4e-4 --num_workers 4 \
           --log_every 50
   ```
   Expect val top-1 to climb past ~40% by epoch 5 (full run reaches >96% by
   E62).

3. **Manifest cache** — on the first run against a new `shards_dir`, rank 0
   builds `<shards_dir>/manifest.json` (per-shard sample counts; ~2–3 min for
   a 1000-shard dataset). Cached for all subsequent runs.

4. **Expected pentagonbox_10x val top-1** (matches saved reference numbers):

   | Variant | best epoch | val top-1 | val top-5 |
   |---|---:|---:|---:|
   | `full`   | 76 | 96.08% | 99.60% |
   | `nosubs` | 62 | 96.08% | 99.60% |

---

## 10. Trained models committed to this repo

| Checkpoint dir | Variant | Data | val top-1 | Notes |
|---|---|---|---:|---|
| [`../checkpoints/pentagonbox_10x_loop_100/`](../checkpoints/pentagonbox_10x_loop_100/) | `full` | `pentagonbox_10x_packed` (all sectors) | 96.08% (E76) | the original production model; used for the published pentagon-box reductions |
| [`../checkpoints/pentagonbox_canon10x_nosubs/`](../checkpoints/pentagonbox_canon10x_nosubs/) | `nosubs` | `pentagonbox_canon10x_packed` (**canonical sectors only**, symmetry-enhanced) | 96.98% (E54, `best_model.pt`) / **97.16%** (E99, `checkpoint_epoch99.pt`) | see that directory's README for loading instructions — it is the `nosubs` class, NOT interchangeable with `full` checkpoints |

The canonical-sector restriction both converges faster (95.3% by E4) and
plateaus ~1 pp higher than the all-sector baseline.
