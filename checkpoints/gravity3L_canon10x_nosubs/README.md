# gravity3L_canon10x_nosubs — trained model (3-loop gravity, symmetry-enhanced)

Action classifier for the **3-loop gravity (GR) topology**, trained
2026-07-15 → 2026-07-17 on `data/gravity3L_canon10x_packed` (canonical
sectors only, zoom 3.50x — see `../../data-gen/README.md` and commit
`48d6b8b` for the dataset recipe).

## The checkpoint

| File | Epoch | val loss | val top-1 | val top-5 |
|------|------:|---------:|----------:|----------:|
| `best_model.pt` | 52 | 0.1077 | 96.47% | 99.69% |

Unlike the pentagonbox canon10x run (where loss-best and top1-best diverged
by ~0.2 pp), gravity3L's val top-1 sat in a tight 96.4–96.5% plateau from
epoch ~45 through 100; the all-time top-1 peak (E67/68, 96.52%) is within
noise of E52. Only the lowest-val-loss checkpoint is committed.

Full run: 100 epochs; final epochs E99/E100 scored 96.47%/96.45% with val
loss drifted up to ~0.120 (mild overfit past E52, same signature as
pentagonbox).

## How to load — `nosubs` variant

The model is `IBPActionClassifierNoSubs` from
[`sailir/classifier_nosubs.py`](../../sailir/classifier_nosubs.py) — **not**
`IBPActionClassifier`. The state_dict has no `subs_enc.*` keys. The model
takes NO substitution input; state = (expression terms, target integral,
sector mask), actions = `(ibp_op, delta)` handles.

```python
import torch
from sailir.classifier_nosubs import IBPActionClassifierNoSubs

ckpt = torch.load(path, map_location=device, weights_only=False)
assert ckpt['args']['model_variant'] == 'nosubs'

model = IBPActionClassifierNoSubs(
    embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
    prime=1009, n_indices=15, n_denominators=10, n_ibp_ops=21,
    # dims verified against Topology.from_dir('topology_input/gravity3L')
    # and the checkpoint's own ckpt['args'].
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

`prime=1009` is REQUIRED (see commit `19e7cb9` for the silent-default bug
this guards against). All hyperparameters are also in `ckpt['args']`.

## Training provenance

- Data: `data/gravity3L_canon10x_packed` (1000 shards, 10.3M train samples/epoch)
- Topology: `topology_input/gravity3L` (n_indices=15, n_denominators=10,
  n_actions=21, 68 masters)
- 100 epochs, 16× A100 DDP (4 nodes), batch 128/rank (effective 2048),
  lr 4e-4 cosine, AdamW wd 1e-5, `--prime 1009`, `--seed 0`
- Launcher: `training/nersc_perlmutter/train_loop.sh` with
  `TOPOLOGY=topology_input/gravity3L SHARDS_DIR=data/gravity3L_canon10x_packed
  MODEL_VARIANT=nosubs` (the launch line documented in commit `f9bac4e`)
- Params: 5,126,153
- Training curve: 93.81% val top-1 after E1 → 96.2% by E19 → plateau
  96.4–96.5% from E45 on. ~265 ms/batch, ~22 min/epoch.
- Ops note: two overnight allocations (Jul 15 21:46 → Jul 16 05:47) ran
  ~50× slow due to a transient filesystem/IO degradation and were
  auto-abandoned by the supervisor; training resumed cleanly from E43
  with no lost epochs.

Numbered per-epoch checkpoints exist only on Perlmutter
(`$SAILIR_DIR/checkpoints/gravity3L_canon10x_nosubs/`).
