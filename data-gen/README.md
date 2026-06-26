# SAILIR — Data-generation phase

This directory is the **data-generation phase** of SAILIR: it produces the
supervised training data for the action-classifier model, for **any integral
topology**. It sits between the topology definition (`../topology_input/`, see
`HOW_TO_DERIVE_FROM_KIRA.md`) and model training (`../training/`); the shared
core library is the `../sailir/` package.

The task SAILIR learns is **"Learning to Unscramble"**: take a random linear
combination of master integrals that has been *scrambled* by applying IBP/LI
identities, and learn to pick, at each step, the identity that reduces it back
toward the masters. This phase manufactures (state → correct-action) examples by
scrambling and then replaying the reduction.

---

## 1. What the two scripts do

| File | What it is | Topology-agnostic? |
|------|------------|--------------------|
| `generate_multisector_data.py` | For each random *scramble*, reduces it to masters with the topology's IBP+LI templates and emits **one training sample per reduction step**. Output: JSONL. | yes (`--topology`) |
| `preprocess_to_tensors.py`      | Converts the raw JSONL into **packed tensor** files (`.pt`) with a train/val/test split, small dtypes, offset-indexed variable-length fields. | yes (`--topology`) |

Both read all topology-specific data (index count, action templates, master
basis) from a `Topology` (`Topology.from_dir(topology_input/<family>)`), so
**nothing here is hardcoded to pentagon-box** — point them at a different
`topology_input/<family>/` and they generate that family's data.

---

## 2. Prerequisites

- **Python** (torch + the `sailir` package):
  `/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python`
- **A prepared topology** `topology_input/<family>/` containing **`IBP`, `LI`,
  `masters`** and **`<FAMILY>_masters_dict.py`** — produced per
  `../topology_input/HOW_TO_DERIVE_FROM_KIRA.md`. The generator needs the
  action templates (`IBP`/`LI`) *and* the master basis (so it knows what to
  scramble back to).
- **The compiled `sailir` Cython kernels** (`python sailir/_setup_cython.py
  build_ext --inplace`); pure-Python fallback works but is slow.
- **A Condor cluster** reachable via `condor_submit`/`condor_q`. The work-dirs
  must be on a shared filesystem (under `data/` on `/het`).

`$SAILIR_DIR` below is the repo root
`/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2`.

---

## 3. The data format

Each line of the raw JSONL is **one reduction step** = one training example:

| field | meaning |
|-------|---------|
| `scramble_id`        | which random scramble this step belongs to |
| `sector_id`          | sector being reduced (bitmask over denominator positions, Kira convention) |
| `sector_mask`        | per-index 0/1 denominator mask |
| `step`               | step index within the scramble |
| `target`, `target_weight` | the integral being reduced at this step and its weight |
| `expr`               | current expression (the state) |
| `subs`               | substitutions already applied |
| `valid_actions`      | the legal IBP/LI actions at this state (the action space) |
| `num_valid_actions`  | its length |
| `chosen_action`, `chosen_action_idx` | **the label** — the action that reduces toward masters |

`preprocess_to_tensors.py` keeps **all** of these, packed into tensors for fast
loading at train time.

---

## 4. The pipeline (for an arbitrary topology)

Two Condor stages. Both launchers carry a small **config block** (`TOPOLOGY`,
`DATASET`, sizes) — edit it and run; nothing else is family-specific.

```
 submit_datagen.sh ──► N workers (datagen_worker.sh ──► generate_multisector_data.py)
        │                                                         │
        ▼                                                         ▼
   data/<DATASET>_raw_jsonl/multisector_data_worker{0..N-1}.jsonl   (raw)
        │
 submit_preprocess_batched.sh ──► N shards (preprocess_shard.sh ──► preprocess_to_tensors.py)
        │                          (Condor file-transfer: in=JSONL, out=shard_<id>/)
        ▼
   data/<DATASET>_packed/shard_{0..N-1}/{train,val,test}.pt        (packed → training)
```

The preprocess works **per shard** (one packed `shard_<id>/` per raw worker
file), so no merge step is required; `merge_outputs.sh` is optional, only for
producing a single raw JSONL.

---

## 5. Quick start

Edit the config block at the top of **both** launchers to the same
`TOPOLOGY` / `DATASET`, then:

```bash
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
cd $SAILIR_DIR

# --- Stage 1: raw JSONL (e.g. 1000 workers x 1000 scrambles = 1M scrambles) ---
#   in submit_datagen.sh:  TOPOLOGY=topology_input/<family>  DATASET=<name>
#                          N_WORKERS=1000  N_SCRAMBLES=1000
bash data-gen/submit_datagen.sh
#   monitor:  condor_q ;  ls data/<name>_raw_jsonl/*.jsonl | wc -l

# --- Stage 2: pack to tensors (same TOPOLOGY/DATASET, N_SHARDS=N_WORKERS) ---
bash data-gen/submit_preprocess_batched.sh
#   result:  data/<name>_packed/shard_*/{train,val,test}.pt   ->  ../training/
```

A single shard can be generated/packed locally to smoke-test before submitting:

```bash
SAILIR_DIR=$SAILIR_DIR TOPOLOGY=topology_input/<family> DATASET=<name> \
  PYTHON=$SAILIR_DIR/../RL_MIR_IBP/conda_env/bin/python \
  bash data-gen/datagen_worker.sh 0 50            # 50 scrambles -> worker0.jsonl
```

---

## 6. Files in this directory

| File | Role |
|------|------|
| `generate_multisector_data.py`   | **core** — scramble → reduce → per-step JSONL samples |
| `preprocess_to_tensors.py`       | **core** — JSONL → packed `.pt` tensors (train/val/test) |
| `submit_datagen.sh`              | Stage 1 launcher (edit config; submits N workers) |
| `datagen_worker.sh`              | Stage 1 Condor worker (one raw shard) |
| `submit_preprocess_batched.sh`   | Stage 2 launcher (batched, throttled submit) |
| `preprocess_shard.sh`            | Stage 2 Condor worker (one packed shard, file-transfer) |
| `merge_outputs.sh`               | optional — concatenate raw JSONL shards into one file |
| `archive/`                       | superseded pentagon-box-specific / experimental variants |
| `logs/`                          | Condor stdout/err/log per job |

---

## 7. Parameters & scaling

Generator knobs (set in `datagen_worker.sh`, passed to
`generate_multisector_data.py`):

- `--n_scrambles` — scrambles per worker. **Total scrambles = `N_WORKERS *
  n_scrambles`.** The pentagon-box production set ("10×") used
  `1000 * 1000 = 1M` scrambles ≈ **13M step-samples**.
- `--min_steps` / `--max_steps` — scramble length range (production: `5`–`25`).
  Longer scrambles → harder, more varied states.
- `--prime` — finite field `GF(p)` for the coefficients (`1009`, matching the
  reduction/topology convention).
- `--start_seed` + the worker's `SEED_STRIDE`/`SEED_OFFSET` — give each worker a
  **disjoint** block of RNG seeds (stride `1e6` ≫ `n_scrambles`). Bump
  `SEED_OFFSET` to make a second dataset that doesn't overlap an existing one.
- advanced filters (off by default): `--filter_lateral`, `--bias-low-s-elim`,
  `--restrict-sectors` — for biasing the sector/step distribution; leave unset
  to match the production "all-sectors, no-bias" recipe.

Preprocess: `--val_split 0.1 --test_split 0.1` (per shard); `--seed` is derived
per worker so the splits differ across shards.

Resources: workers request `1 CPU / 6 GB` each. The preprocess uses **Condor
file transfer** (input JSONL → scratch, packed `shard_<id>/` → packed dir) and
is submitted in batches (`BATCH_SIZE`/`LOW_WATER`) to avoid hammering NFS with
1000 simultaneous python starts.

---

## 8. Sanity checks before training

1. `ls data/<name>_raw_jsonl/*.jsonl | wc -l` == `N_WORKERS` (all workers wrote).
2. `wc -l data/<name>_raw_jsonl/multisector_data_worker0.jsonl` — nonzero, and
   roughly `n_scrambles * (avg steps per scramble)` samples.
3. `ls data/<name>_packed/shard_*/train.pt | wc -l` == `N_SHARDS` (all packed).
4. Spot-load one shard in python and confirm the field set matches §3 and the
   index length equals the topology's `n_indices`.
