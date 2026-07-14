# pentagonbox_canon10x_nosubs — trained model (symmetry-enhanced retrain)

Action classifier for the **symmetry-enhanced** reduction stack, trained
2026-07-12 → 2026-07-14 on `data/pentagonbox_canon10x_packed` (canonical
sectors only — see `../../data-gen/README.md` for the dataset recipe).

## The two checkpoints

| File | Epoch | val loss | val top-1 | val top-5 | Which to use |
|------|------:|---------:|----------:|----------:|--------------|
| `best_model.pt` | 54 | 0.0934 | 96.98% | 99.68% | lowest val loss (best-calibrated probabilities) |
| `checkpoint_epoch99.pt` | 99 | 0.1000 | **97.16%** | 99.67% | peak top-1 accuracy |

Val loss bottomed at E54 and drifted up while top-1 kept creeping — mild
overfit. Which is better for beam-search reduction is untested; the
difference is likely negligible. Baseline for comparison: the all-sector
`pentagonbox_10x` runs plateaued at 96.08% top-1.

## How to load — IMPORTANT: this is the `nosubs` variant

The model is `IBPActionClassifierNoSubs` from
[`sailir/classifier_nosubs.py`](../../sailir/classifier_nosubs.py) — **not**
the `IBPActionClassifier` (`full`) class that loads the older
`pentagonbox_10x_loop_100/best_model.pt`. The state_dict has no `subs_enc.*`
keys; loading it into the `full` class fails with missing-key errors.

The model takes NO substitution input of any kind. State = (expression
terms, target integral, sector mask); actions = `(ibp_op, delta)` handles.
The `forward()` still *accepts* the `sub_*` tensors for call-site parity
with `full`, but discards them.

```python
import torch
from sailir.classifier_nosubs import IBPActionClassifierNoSubs

ckpt = torch.load(path, map_location=device, weights_only=False)
assert ckpt['args']['model_variant'] == 'nosubs'   # auto-select key

model = IBPActionClassifierNoSubs(
    embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
    prime=1009, n_indices=11, n_denominators=8, n_ibp_ops=18,
    # dims verified against Topology.from_dir('topology_input/pentagonbox')
    # and the checkpoint's own ckpt['args'].
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

All architecture hyperparameters are also recorded in `ckpt['args']` for
fully programmatic construction.

## Training provenance

- Data: `data/pentagonbox_canon10x_packed` (1000 shards, 10.9M train samples/epoch)
- 100 epochs, 16× A100 DDP (4 nodes), batch 128/rank (effective 2048), lr 4e-4
  cosine, AdamW wd 1e-5, `--prime 1009`, `--seed 0`
- Launcher: `training/nersc_perlmutter/train_loop.sh` with
  `SHARDS_DIR=data/pentagonbox_canon10x_packed MODEL_VARIANT=nosubs`
- Code at training time: branch `symmetry-enhanced`, commit `b6f6825`
- Params: 4,898,449

Numbered per-epoch checkpoints for other epochs exist only on Perlmutter
(`$SAILIR_DIR/checkpoints/pentagonbox_canon10x_nosubs/`); only the two above
are committed.
