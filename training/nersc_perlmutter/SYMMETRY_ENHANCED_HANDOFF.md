# Symmetry-Enhanced Training — Perlmutter handoff

Retrain the SAILIR action-classifier on the **symmetry-enhanced** dataset and
compare it to the baseline `pentagonbox_10x` model. This doc is the training-side
(Perlmutter) half; data-gen + evaluation happen on het.

Branch: **`symmetry-enhanced`** (has the NERSC package *and* the data-gen change).

---

## 1. What changed (one sentence)

The data-gen now scrambles **only one representative per clean-symmetry orbit** —
**174 canonical sectors instead of 255** — so the training set is "zoomed in":
symmetry-equivalent sectors are no longer generated as redundant relabelings.
(Pentagon-box zoom is modest, ~1.47×; see §6 on expectations.)

The canonicalization is a genuine loop-relabeling (`_target_key`-ordered clean
orbit, `reduction/canonical_rep.py`), **not** the affine relations — those stay the
job of the inference-time routing.

---

## 2. The split (who does what)

| machine | role | in this build |
|---|---|---|
| **het** | data-gen + reduction | generates the symmetry-enhanced packed dataset, rsyncs it here; builds/validates the inference-side canonicalization (the pairing, §5) |
| **Perlmutter** | training | **this doc** — train on the packed dataset |
| **het** | eval | A/B the retrained model (+ canonicalization) vs baseline |

---

## 3. Prerequisites on Perlmutter

```bash
cd /global/homes/d/dshih/m4539_d/SAILIR_phase2
git fetch origin && git checkout symmetry-enhanced && git pull
```

The symmetry-enhanced packed dataset must be present at
`data/pentagonbox_sym_packed/shard_*/{train,val,test}.pt` (rsync'd from het —
het generates it with `generate_multisector_data.py --restrict-sectors-file
results/canonical_sectors_tkey.txt`, then `preprocess_to_tensors.py`, exactly
like the `10x` pipeline but with the sector restriction).

Sanity: `ls data/pentagonbox_sym_packed/shard_*/train.pt | wc -l` should equal the
number of raw workers, same as the `10x` set.

---

## 4. Training

Identical to the `10x` run (README) except pointed at the symmetry dataset via the
new `SHARDS_DIR` override:

```bash
# inside a tmux session on the login node, after salloc (see README §2):

# smoke test FIRST — verifies DDP boundary on the new shards
SMOKE=1 SHARDS_DIR=data/pentagonbox_sym_packed \
  OUTPUT_DIR=checkpoints/pentagonbox_sym_smoke \
  bash training/nersc_perlmutter/allocation.sh

# full run (20 epochs, auto_resume across 4-hour allocations)
SHARDS_DIR=data/pentagonbox_sym_packed \
  OUTPUT_DIR=checkpoints/pentagonbox_sym \
  bash training/nersc_perlmutter/allocation.sh
```

`SHARDS_DIR` and `OUTPUT_DIR` are the only new knobs — DDP rendezvous, tmux,
`--auto_resume`, the supervisor loop (`train_loop.sh`), batch/lr/shard defaults are
all unchanged from the main [README](README.md). Use a supervisor loop the same way,
but with `OUTPUT_DIR=checkpoints/pentagonbox_sym` on **both** the supervisor and the
allocation.

Output: `checkpoints/pentagonbox_sym/`, `logs/pentagonbox_10x_full_<ts>.log`.

---

## 5. CRITICAL — the inference pairing

A model trained on the canonical-only dataset has **only ever seen canonical-sector
integrals**. At inference it **must** be run with matching canonicalization: every
integral mapped to its `_target_key` canonical representative
(`reduction/canonical_rep.py`) *before* the model scores it. Without that, the model
meets out-of-distribution sectors and degrades.

That inference-side canonicalization is built and validated **on het** — the
retrained checkpoint is only half the pipeline. Do not benchmark the symmetry
checkpoint with the plain (non-canonicalizing) reducer and conclude anything.

---

## 6. Validation & honest expectations

- **Smoke test** must pass (5 epochs, exercises the train→val boundary + post-epoch
  all-reduce) before committing a 4-hour allocation.
- **Full run:** `val_nll` should at least track the baseline `pentagonbox_10x`;
  the hope is it beats it slightly (fewer, more concentrated sectors → cleaner
  signal). If it is clearly *worse*, suspect a data/pairing mismatch, not the model.
- **The real test is het's A/B**, not `val_nll`: reduce the benchmark integrals with
  `checkpoints/pentagonbox_sym` + canonicalization and compare **total worker CPU**
  (not just worker count — see `notes/symmetry_inference_routing.tex` for why) vs the
  baseline model.

**Expectations, stated plainly:** pentagon-box symmetry zoom is modest (1.47×
sectors), so do not expect a dramatic pentagon-box win. The value of this build is
the **pipeline + validating the mechanism end-to-end**. The payoff scales with how
symmetric the topology is; the long-term target is the **gravity loop integrals**
(expected much richer symmetry). Before investing there, re-run
`reduction/build_canonical_sectors_tkey.py` on the gravity topology and check the
zoom + `P_S` coverage — if it is high (say ≥ 2–3× sector zoom), this whole pipeline
pays off there in a way it only marginally does for pentagon-box.
