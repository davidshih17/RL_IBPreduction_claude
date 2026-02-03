# IBP Reduction with ML-Guided Beam Search

This repository contains code for ML-guided Integration-by-Parts (IBP) reduction of Feynman integrals. We train a transformer-based classifier to guide beam search through the space of IBP identities, achieving fast and correct reductions.

## Key Results

### Latest: Async Parallel Reduction with Condor

| Integral | Weight | Time (Sequential) | Time (Parallel) | Speedup | Masters |
|----------|--------|-------------------|-----------------|---------|---------|
| `I[1,1,1,1,1,1,-3]` | (6,3) | 73 min | 12 min | 6x | 16 |
| `I[3,2,1,3,2,2,-6]` | (13,6) | ~20 hr | **115 min** | **~10x** | 16 |

**Async parallelization features:**
- Distributes one-step reductions across Condor CPU nodes
- Memoization cache avoids redundant work (~55k cache hits)
- Straggler detection: jobs >30 min are killed and resubmitted with 8 CPUs
- Paper-masters-only mode: reduce to exactly 16 paper masters from arXiv:2502.05121

See [docs/parallelization.md](docs/parallelization.md) for detailed documentation.

### Previous: V14 with Checkpoint/Resume

| Integral | Weight | Sectors | Steps | Time | Masters |
|----------|--------|---------|-------|------|---------|
| `I[1,1,1,1,1,1,-3]` | (6,3) | 45 | 1,416 | 353s | 38 |
| `I[3,2,1,3,2,2,-6]` | (13,6) | 62 | 46,345 | ~19.7 hr | 63 |

**V14 features:**
- Checkpoint after each sector completion for crash recovery
- Resume from checkpoint if interrupted
- Beam restart strategy: prune to best state after each weight improvement

### V5 Hierarchical Reduction

- Reduced `I[2,0,2,0,1,1,0]` (sector 53) to 4 master integrals in **~5 minutes**
- Results match Kira exactly
- ~44x faster than AIR (which took ~3.7 hours)

## Architecture Overview

The system uses a transformer-based action classifier (V5) with:

1. **Expression Encoder**: Self-attention over expression terms with a special [TARGET] token
2. **Full Substitution Encoder**: Encodes complete substitution history (key + all replacement terms)
3. **Cross-Attention Scorer**: Actions attend to expression terms for context-aware scoring

See [docs/progress_summary_v5_hierarchical.md](docs/progress_summary_v5_hierarchical.md) for detailed architecture documentation.

## Repository Structure

```
ibp-neural-reduction/
├── models/
│   ├── classifier_v5.py          # V5 model architecture
│   ├── classifier_v4_target.py   # V4 target classifier
│   ├── classifier_v3.py          # Cross-attention scorer
│   ├── classifier_v2.py          # Base encoder components
│   ├── classifier_v1.py          # Collate functions
│   └── ibp_env.py                # IBP environment with caching + paper-masters-only mode
├── scripts/
│   ├── training/
│   │   └── train_classifier_v5.py    # Training script
│   └── eval/
│       ├── hierarchical_reduction_async.py   # Async parallel with Condor (recommended)
│       ├── reduce_integral_onestep_worker.py # Condor worker for one-step reductions
│       ├── hierarchical_reduction_v14.py     # V14: checkpoint/resume (sequential)
│       ├── beam_search_classifier_v11.py     # V11: beam restart strategy
│       ├── beam_search_classifier_v5.py      # V5: optimized beam search
│       └── replay_reduction_path.py          # Replay saved reductions
├── results/
│   └── reduction_321322m6_async_papermasters.pkl  # Async reduction (115 min)
├── logs/
│   └── async_321322m6_papermasters.log  # Async run log
├── checkpoints/
│   └── classifier_v5/
│       └── best_model.pt         # Trained model checkpoint
└── docs/
    ├── parallelization.md                    # Parallel reduction documentation
    └── progress_summary_v5_hierarchical.md   # Technical documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/IBPreduction.git
cd IBPreduction

# Install dependencies
pip install torch numpy
```

## Usage

### Running Async Parallel Reduction (Recommended)

For large reductions, use the async parallel approach with Condor:

```bash
python -u scripts/eval/hierarchical_reduction_async.py \
    --integral 3,2,1,3,2,2,-6 \
    --output results/reduction.pkl \
    --work-dir /scratch/ibp_async \
    --beam_width 20 \
    --prime 1009 \
    --paper-masters-only \
    --straggler-timeout 1800 \
    --straggler-cpus 8
```

**Async Arguments:**
- `--integral`: Starting integral indices (comma-separated)
- `--output`: Output pickle file for final reduction
- `--work-dir`: Working directory for Condor job files and intermediate results
- `--paper-masters-only`: Reduce to 16 paper masters only (no corner integrals)
- `--straggler-timeout`: Seconds before resubmitting slow jobs (default: 1800 = 30 min)
- `--straggler-cpus`: CPUs for resubmitted stragglers (default: 8)
- `--beam_width`: Beam width for search (default: 20)
- `--prime`: Prime for modular arithmetic (default: 1009)

**Requirements:** HTCondor cluster with CPU nodes available.

### Running V14 Hierarchical Reduction (Sequential)

```bash
# Fresh start with checkpointing
python -u scripts/eval/hierarchical_reduction_v14.py \
    --integral 1,1,1,1,1,1,-3 \
    --output results/reduction_111111m3.pkl \
    --checkpoint-dir checkpoints/reduction_111111m3 \
    --beam_width 20 \
    --max_steps 500 \
    --device cuda

# Resume from checkpoint after interruption
python -u scripts/eval/hierarchical_reduction_v14.py \
    --resume checkpoints/reduction_111111m3
```

**V14 Arguments:**
- `--integral`: Starting integral indices (comma-separated)
- `--output`: Output pickle file for reduction path
- `--checkpoint-dir`: Directory to save checkpoints after each sector
- `--resume`: Resume from checkpoint directory
- `--beam_width`: Beam width for search (default: 20)
- `--max_steps`: Maximum steps per beam restart (default: 500)
- `--device`: `cuda` or `cpu`

### Replaying a Saved Reduction

```bash
python -u scripts/eval/replay_reduction_path.py \
    --path results/reduction_111111m3_v14.pkl \
    --prime 10007
```

### Running V5 Hierarchical Reduction

```bash
python -u scripts/eval/hierarchical_reduction_v5.py \
    --integral 2,0,2,0,1,1,0 \
    --checkpoint checkpoints/classifier_v5/best_model.pt \
    --beam_width 20 \
    --max_steps 100 \
    --filter_mode subsector \
    --device cuda \
    -v
```

**Arguments:**
- `--integral`: Starting integral indices (comma-separated)
- `--checkpoint`: Path to model checkpoint
- `--beam_width`: Beam width for search (default: 20)
- `--max_steps`: Maximum steps per sector (default: 100)
- `--filter_mode`: Sector filtering (`subsector`, `higher_only`, `none`)
- `--device`: `cuda` or `cpu`
- `-v`: Verbose output

### Running Single-Sector Beam Search

```bash
python -u scripts/eval/beam_search_classifier_v5.py \
    --integral 2,0,1,0,2,0,0 \
    --checkpoint checkpoints/classifier_v5/best_model.pt \
    --beam_width 20 \
    --max_steps 100 \
    --device cuda
```

### Training

```bash
python -u scripts/training/train_classifier_v5.py \
    --data_dir data/multisector_tensors_v2 \
    --output_dir checkpoints/classifier_v5 \
    --epochs 30 \
    --batch_size 256 \
    --lr 0.0004 \
    --device cuda
```

**Training data format:** Packed tensor format with:
- Expression integrals and coefficients
- Full substitution structure (key + replacement terms)
- Target integral
- Valid actions and labels

## Model Details

**IBPActionClassifierV5:**
- Parameters: 7.7M
- Embed dim: 256
- Attention heads: 4
- Expression encoder layers: 2
- Substitution encoder layers: 2
- Cross-attention layers: 2

**Training:**
- Dataset: 946K training samples across all 63 sectors
- Best validation accuracy: 90.77% (epoch 22)
- Prime for modular arithmetic: 1009

## IBP Environment

The IBP environment (`models/ibp_env.py`) handles:
- Loading IBP and LI equation templates
- Evaluating IBP equations at specific seeds
- Applying substitutions to expressions
- Enumerating valid actions with sector filtering
- Caching raw equations for performance

**Key optimization:** Cached `get_raw_equation` results provide 3-10x speedup for action enumeration.

## Hierarchical Reduction Strategy

1. Identify highest-level sector with non-master integrals
2. Run beam search to eliminate all non-masters in that sector
3. Move to next highest sector
4. Repeat until only master integrals remain

This processes sectors in order: 63 → 62 → ... → lower sectors

### V11+ Beam Restart Strategy

Instead of running beam search until completion or timeout, V11+ uses a **restart strategy**:

1. Run beam search until weight improves
2. Prune beam to single best state
3. Restart beam search from that state
4. Repeat until sector is fully reduced

This prevents the beam from getting stuck with suboptimal states and allows much deeper reductions.

### V14 Checkpointing

V14 saves a checkpoint after each sector completion:
- Checkpoint includes: current expression, accumulated path, sector stats
- Can resume from checkpoint if interrupted
- Human-readable summary file for monitoring progress

## Citation

If you use this code, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]
