# IBP Reduction with ML-Guided Beam Search

This repository contains code for ML-guided Integration-by-Parts (IBP) reduction of Feynman integrals. We train a transformer-based classifier to guide beam search through the space of IBP identities, achieving fast and correct reductions.

## Key Results

- Reduced the two-loop triangle-box integral `I[2,0,2,0,1,1,0]` (sector 53) to 4 master integrals in **~5 minutes**
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
IBPreduction/
├── models/
│   ├── classifier_v5.py          # V5 model architecture
│   ├── classifier_v2.py          # Base encoder components
│   └── ibp_env.py                # IBP environment with caching
├── scripts/
│   ├── training/
│   │   └── train_classifier_v5.py    # Training script
│   └── eval/
│       ├── beam_search_classifier_v5.py      # Optimized beam search
│       └── hierarchical_reduction_v5.py      # Hierarchical reduction
├── checkpoints/
│   └── classifier_v5/
│       └── best_model.pt         # Trained model checkpoint
├── data/
│   └── multisector_tensors_v2/   # Training data (not included)
└── docs/
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

### Running Hierarchical Reduction

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

This processes sectors in order: 53 → 52 → 49 → 37 → 21 → lower sectors

## Citation

If you use this code, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]
