# SAILIR

**S**elf-supervised **AI** for **L**oop **I**ntegral **R**eduction.

Code, trained model, and benchmark data accompanying:

> **Learning to Unscramble Feynman Loop Integrals with SAILIR**
> David Shih, 2026. arXiv:2604.05034.

## Topologies

Phase-2 makes the pipeline **topology-agnostic**: every script (data-gen,
preprocess, train, eval) takes a `--topology topology_input/<family>/`
argument that selects the integral family at runtime. The repo ships with
two configurations:

- `topology_input/trianglebox/` — 2-loop triangle-box (6 propagators + 1 ISP),
  the published phase-1 family. Used by all examples below unless noted.
- `topology_input/pentagonbox/` — 2-loop pentagon-box, TA family from Kira
  (8 propagators + 3 ISPs). See `topology_input/HOW_TO_DERIVE_FROM_KIRA.md`
  for how to derive a new topology's inputs.
- `topology_input/gravity3L/` — 3-loop gravity (post-Minkowskian potential)
  family, FIRE family 40 (10 propagators incl. linear/eikonal + 5 ISPs).

**Production reduction runs** use the **symmetry-enhanced general-topology
pipeline** — sector canonicalization, the sector-senior total order, canonical
masters, and a general, numerically-gated symmetry engine — documented in
[`reduction/README.md`](reduction/README.md) (§4b there is the entry point;
the examples below reproduce the paper's plain IBP+LI runs).

Each topology directory contains:

```
integralfamilies.yaml   propagators + ISPs + symmetry classes
kinematics.yaml         kinematic invariants (d, s_ij, ...) + their finite-field values
IBP                     IBP identity templates (shift, coefficient)
LI                      Lorentz-invariance identities
masters                 Kira's master basis
```

## Repository layout

```
sailir/
  topology.py            Topology dataclass; init_from_topology() configures the module
  ibp_env.py             IBP environment (identities, action enumeration); topology-driven
  classifier.py          Action classifier; encoder dims taken from the topology
  symmetries.py          Sector-symmetry library (standalone, optional)
scripts/
  data_gen/              Self-supervised trajectory generation + JSONL→tensor packing
                         (now supports multi-file --input and per-worker sharding)
  train/train_classifier.py
                         Cross-entropy training with topology-aware collate
  eval/                  Hierarchical async orchestrator + single-integral worker + replay
  kira/                  Kira inputs for the comparison side of Fig. 3
  plots/                 Fig. 2 and Fig. 3
topology_input/<family>/ Per-topology config (see above)
checkpoints/best_model.pt
                         Published phase-1 trained model (trianglebox)
results/                 Benchmark CSVs and training log
```

## Requirements

- Python 3.10+
- PyTorch, NumPy, Matplotlib
- HTCondor (for §1 data-gen and §4 benchmark)
- Kira + Fermat (for §6 only)

```bash
pip install torch numpy matplotlib
```

## Quick start

Reduce a single trianglebox integral with the published checkpoint:

```bash
python scripts/eval/onestep_worker.py \
    --topology topology_input/trianglebox \
    --integral=2,1,2,1,2,2,-4 \
    --model-checkpoint checkpoints/best_model.pt \
    --output reduction.pkl \
    --beam_width 20 \
    --prime 1009 \
    --paper-masters-only -v

python scripts/eval/replay_reduction_path.py --path reduction.pkl
```

## End-to-end reproduction (trianglebox)

### 1. Generate training data

100 Condor workers × 1000 scrambles each (`--max_steps 25`). ≈18% of attempts
are discarded as `SKIPPED_VANISHING`; the remaining ≈82 000 trajectories yield
the $8\times10^4$ trajectories / $1.06\times10^6$ samples reported in paper §IV.B.

```bash
export SAILIR_DIR=$(pwd)
export PYTHON=$(which python)

bash data-gen/submit_datagen.sh 100 1000
condor_submit data-gen/datagen_job_custom.jdl
# ... wait for all 100 jobs to finish ...
bash data-gen/merge_outputs.sh data/raw_jsonl/
```

### 2. Pack JSONL into tensors

```bash
python data-gen/preprocess_to_tensors.py \
    --topology   topology_input/trianglebox \
    --input      data/raw_jsonl/multisector_training_data.jsonl \
    --output_dir data/multisector/
```

Defaults reproduce the paper's preprocessing: `--val_split 0.1`, `--test_split 0.1`, `--seed 42`. Produces `train.pt` (≈80%), `val.pt` (≈10%), `test.pt` (≈10%).

`--input` accepts multiple files now (e.g. all per-worker JSONLs) and
streams them in sequence, so no concatenation step is needed.

### 3. Train

```bash
python training/train_classifier.py \
    --topology   topology_input/trianglebox \
    --data_dir   data/multisector/ \
    --output_dir checkpoints/ \
    --epochs 30 --batch_size 256 --lr 4e-4 --prime 1009 --device cuda

python scripts/plots/plot_training_curve.py --log results/training_log/train.log
```

### 4. Reduce the 16 benchmark integrals

```bash
INTEGRALS=( "2,1,2,1,2,2,-4"  "1,1,2,2,1,3,-5"  "1,1,3,2,2,1,-6"  "2,3,1,1,2,1,-7"
            "2,2,2,1,1,3,-4"  "1,1,2,3,2,2,-5"  "1,4,2,1,2,1,-6"  "2,1,1,2,3,2,-7"
            "2,3,1,3,1,2,-4"  "1,2,2,2,1,4,-5"  "3,2,3,2,1,1,-6"  "3,1,1,1,1,5,-7"
            "2,3,3,3,1,1,-4"  "2,2,3,3,2,1,-5"  "3,2,1,3,2,2,-6"  "2,2,3,3,1,2,-7" )

mkdir -p logs results work-dir
for I in "${INTEGRALS[@]}"; do
    label=$(echo "$I" | tr ',' '_' | tr '-' 'm')
    python -u scripts/eval/hierarchical_reduction.py \
        --topology topology_input/trianglebox \
        --integral=$I --output results/reduction_${label}.pkl \
        --work-dir work-dir/${label} \
        --model-checkpoint checkpoints/best_model.pt \
        --beam_width 20 --prime 1009 --paper-masters-only --beam-sort mixed \
        > logs/async_${label}.log 2>&1 &
done
wait
```

### 5. Plot

```bash
python scripts/eval/collect_benchmark_results.py \
    --logdir logs/ --resultdir results/ --scratchdir work-dir/

python scripts/plots/plot_benchmark_comparison.py \
    --sailir-csv results/benchmark_mixed_summary_v13.csv \
    --kira-csv   results/kira_benchmark.csv \
    --out        results/benchmark_comparison.pdf
```

`results/sailir_benchmark.csv` is the CSV used in the paper; pass it to
`--sailir-csv` to regenerate Fig. 3 verbatim.

### 6. Kira benchmark (optional)

```bash
export KIRA=/path/to/kira/bin/kira
export FERMATPATH=/path/to/fermat/fer64
cd scripts/kira/ && ./run_benchmark_s7.sh
```

## Pentagon-box (TA family)

The pentagon-box pipeline mirrors the trianglebox flow with the
`topology_input/pentagonbox` directory and the dedicated launcher set:

### Data-gen

```bash
export SAILIR_DIR=$(pwd)
export PYTHON=$(which python)

condor_submit data-gen/datagen_pentagonbox.jdl     # 100 × 1000 (= 1×)
# or:
condor_submit data-gen/datagen_pentagonbox_10x.jdl # 1000 × 1000 (= 10×)
```

Output JSONLs land in `data/pentagonbox_raw_jsonl/` (or
`data/pentagonbox_10x_raw_jsonl/` for the 10× run; seeds are offset to keep
them disjoint from the 1× set).

### Sharded preprocess (large datasets)

For the 10× dataset, packing all ~13M samples in a single Python process
would exhaust memory. Each worker JSONL is preprocessed independently into
its own packed shard:

```bash
bash data-gen/submit_preprocess_pentagonbox_10x_batched.sh
```

This submits the 1000 shards in throttled batches via Condor file transfer
(input staged into `$_CONDOR_SCRATCH_DIR`, output transferred back to
`data/pentagonbox_10x_packed/shard_<N>/`). Each shard packs to
`train.pt + val.pt + test.pt` (~115 MB total). At train time, wrap the
shards in `torch.utils.data.ConcatDataset` with `torch.load(..., mmap=True)`
to keep peak RAM bounded.

### Train

```bash
condor_submit training/train_pentagonbox.sub
```

Or directly:

```bash
python training/train_classifier.py \
    --topology   topology_input/pentagonbox \
    --data_dir   data/pentagonbox_packed/ \
    --output_dir checkpoints/pentagonbox/ \
    --epochs 30 --batch_size 128 --lr 4e-4 --prime 1009 --device cuda
```

### Reduce

Same as trianglebox, but with `--topology topology_input/pentagonbox` and an
11-index integral, e.g.:

```bash
python scripts/eval/onestep_worker.py \
    --topology topology_input/pentagonbox \
    --integral=1,0,1,0,1,1,-1,0,0,0,-1 \
    --model-checkpoint checkpoints/pentagonbox/best_model.pt \
    --output reduction.pkl \
    --beam_width 20 --prime 1009 -v
```

## Adding a new topology

See `topology_input/HOW_TO_DERIVE_FROM_KIRA.md`. In brief, run Kira to
extract: integralfamilies.yaml, kinematics.yaml, IBP+LI identity templates,
and the master basis. Drop them into `topology_input/<family>/` and every
script accepts that path via `--topology`.

## Citation

```bibtex
@article{Shih:2026jfe,
    author = "Shih, David",
    title = "{Learning to Unscramble Feynman Loop Integrals with SAILIR}",
    eprint = "2604.05034",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2026"
}
```

## License

MIT.
