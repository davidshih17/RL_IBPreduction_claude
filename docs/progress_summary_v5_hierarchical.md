# IBP Reduction with ML-Guided Hierarchical Beam Search

## Summary

We successfully built an ML-guided IBP (Integration-by-Parts) reduction system that matches the output of professional reduction software (Kira) while being significantly faster than alternative approaches (AIR).

### Key Achievement

Reduced the two-loop triangle-box integral `I[2,0,2,0,1,1,0]` (sector 53) to its 4 master integrals in **302 seconds (~5 minutes)**, producing results that exactly match Kira.

| Method | Time | Final Masters |
|--------|------|---------------|
| Kira | - | 4 masters (reference) |
| Our V5 | 302s (~5 min) | 4 masters (matches Kira) |
| AIR | 13,193s (~3.7 hr) | 5 "masters" (corner basis) |

Our approach is **~44x faster than AIR** and produces the correct minimal master basis.

---

## Final Result

Starting integral: `I[2,0,2,0,1,1,0]` (weight 6, sector 53)

Reduced to 4 paper masters:
- `107 * I[1,-1,1,0,1,1,0]` (sector 53 master)
- `303 * I[1,0,1,0,1,1,0]` (sector 53 master)
- `915 * I[1,0,1,0,0,1,0]` (sector 37 master)
- `752 * I[1,0,1,0,1,0,0]` (sector 21 master)

All coefficients are mod 1009.

---

## Data Generation

### Multisector Scramble Approach

**File:** `scripts/data_gen/generate_multisector_data.py`

Training data is generated via "scrambling" - reverse reduction starting from master integrals:

1. **Start from masters:** Begin with a master integral (from the 16 paper masters or sector corners)
2. **Random IBP applications:** Apply random IBP identities to increase expression complexity
3. **Record reduction steps:** Each scramble step becomes a training example when reversed

**Key constraints during scrambling:**
- Only apply IBP actions that don't introduce higher-sector integrals
- Filter actions to stay within the target sector and its subsectors
- Ensures training data reflects valid reduction trajectories

**Sector coverage:**
- Covers all 63 non-trivial sectors (sector_id 1-63)
- Uses paper masters for 13 sectors that have them
- Uses corner integrals for remaining sectors

### Data Format

Each training sample contains:
- `sector_mask`: 6-bit binary encoding of target sector
- `expr`: Current expression (list of integrals + coefficients)
- `subs`: Accumulated substitutions (full replacement expressions)
- `target_integral`: The integral being eliminated
- `valid_actions`: All valid (ibp_op, delta) pairs
- `label`: Index of the chosen action (the one that leads to reduction)

### Dataset Statistics

**Training data:** `data/multisector_tensors_v2/`
- Train: 946,168 samples (3.8 GB)
- Validation: 118,271 samples (480 MB)
- Test: ~118K samples (480 MB)

---

## Training

### Training Configuration

**File:** `scripts/training/train_classifier_v5.py`

```
epochs: 30
batch_size: 256
learning_rate: 0.0004
weight_decay: 1e-5
optimizer: AdamW
```

### Model Architecture

```
embed_dim: 256
n_heads: 4
n_expr_layers: 2
n_cross_layers: 2
n_subs_layers: 2
total_parameters: 7,696,709
```

### Training Results

- **Best checkpoint:** Epoch 22
- **Validation accuracy:** 90.77% (top-1)
- **Validation top-5 accuracy:** ~98%
- **Training time:** ~800s per epoch on GPU

### Key V5 Features

1. **Target integral as input:** Model receives the specific integral being eliminated
2. **Full substitution encoding:** Each substitution includes:
   - Key integral (7 indices)
   - Complete replacement expression (up to 20 terms with integrals + coefficients)
3. **Sector conditioning:** Sector mask provided to model for context

---

## System Architecture

### 1. Model: IBPActionClassifierV5

**File:** `models/classifier_v5.py`

The model scores candidate IBP actions given the current reduction state. It uses a transformer-based architecture with multiple specialized encoders and cross-attention for action scoring.

#### High-Level Data Flow

```
Inputs:
  - Expression (integrals + coefficients)
  - Target integral (the one being eliminated)
  - Substitution history (key integrals + their replacements)
  - Sector mask (6-bit encoding)
  - Candidate actions (ibp_op + delta)

                    ┌─────────────────────────────────────────┐
                    │     TransformerExpressionEncoder        │
                    │  [CLS] [TARGET] term1 term2 ... termN   │
                    │         ↓ Self-Attention (2 layers)     │
                    └─────────────────────────────────────────┘
                              ↓                    ↓
                         cls_pooled          expr_terms (per-term)
                              │                    │
    ┌──────────────┐          │                    │
    │FullSubsEnc   │          │                    │
    │  (2 layers)  │→ subs_emb│                    │
    └──────────────┘          │                    │
                              │                    │
    ┌──────────────┐          │                    │
    │ SectorEnc    │→sector_emb                    │
    └──────────────┘          │                    │
                              ↓                    │
                    ┌─────────────────┐            │
                    │  State Combine  │            │
                    │ (4*256 → 256)   │            │
                    └────────┬────────┘            │
                             │                     │
                         state_emb                 │
                             │                     │
                             ↓                     ↓
                    ┌─────────────────────────────────────────┐
                    │        CrossAttentionScorer             │
                    │  actions attend to expr_terms           │
                    │         ↓ Cross-Attention (2 layers)    │
                    │         ↓ Score each action             │
                    └─────────────────────────────────────────┘
                                      ↓
                              logits (per action)
```

#### Component Details

**1. TransformerExpressionEncoderWithTarget**
- Encodes expression terms AND the target integral
- Sequence format: `[CLS] [TARGET] term1 term2 ... termN`
- Each term = IntegralEncoder(indices) + CoefficientEncoder(coeff)
- 2-layer Transformer with self-attention
- Outputs:
  - `cls_pooled`: Global expression representation
  - `target_pooled`: Target-aware representation
  - `expr_terms`: Per-term embeddings (used for cross-attention)

**2. FullSubstitutionEncoder**
- Encodes the complete substitution history
- Each substitution = (key_integral, [(repl_int1, coeff1), (repl_int2, coeff2), ...])
- Process:
  1. Encode key integral
  2. Encode each replacement term (integral + coefficient)
  3. **Attention pool** replacement terms into single embedding
  4. Combine key + pooled_replacement via MLP
  5. Add positional encoding (substitution order matters)
  6. 2-layer Transformer self-attention over substitution sequence
  7. **Attention pool** to final embedding
- This is the key V5 innovation: previous versions only encoded the key, missing the replacement structure

**3. SectorEncoder**
- Encodes 6-bit sector mask
- Each bit position gets its own embedding
- Concatenate + project to embed_dim

**4. ActionEncoder**
- Encodes candidate actions as (ibp_op, delta)
- ibp_op: Which IBP operator (0-7 for IBP, 8 for LI)
- delta: Offset from target to seed integral

**5. CrossAttentionScorer**
- Scores actions by attending to expression terms
- Process:
  1. Project state, actions, and expr_terms
  2. **Cross-attention (2 layers)**: Each action attends to all expression terms
  3. Concatenate attended_action with state
  4. MLP scorer → logit per action
- This allows actions to "look at" which expression terms they would affect

#### Architecture Summary

| Component | Type | Layers | Attention |
|-----------|------|--------|-----------|
| Expression Encoder | Transformer | 2 | Self-attention over [CLS][TARGET][terms] |
| Substitution Encoder | Transformer | 2 | Self-attention over substitution sequence |
| Replacement Pooling | Attention | 1 | Pool replacement terms per substitution |
| Cross-Attention Scorer | Cross-Attn | 2 | Actions attend to expression terms |

**Total parameters:** 7,696,709

**Checkpoint:** `checkpoints/classifier_v5/best_model.pt` (epoch 22, val acc: 90.77%)

### 2. Beam Search with Optimizations

**File:** `scripts/eval/beam_search_classifier_v5.py`

**Key optimizations:**

1. **Batched Model Inference (P2 optimization: ~50x speedup)**
   - Prepare all action candidates as numpy arrays
   - Single batched forward pass through the model
   - Eliminates per-action inference overhead

2. **Cached `get_raw_equation` Results (P1 optimization: ~3-10x speedup)**
   - IBP equation generation is expensive (sympy operations)
   - Cache results in `env._raw_eq_cache` dictionary
   - Key: `(ibp_op, seed)` tuple
   - Reuse across beam states with shared substitution history

**Implementation in `models/ibp_env.py`:**
```python
def get_valid_actions_cached(self, target, subs, filter_mode='subsector'):
    """Get list of valid actions for target, using cached raw equations."""
    return enumerate_valid_actions_cached(
        target, subs, self.ibp_t, self.li_t, self.shifts, filter_mode,
        self._raw_eq_cache
    )
```

### 3. Hierarchical Reduction Strategy

**File:** `scripts/eval/hierarchical_reduction_v5.py`

**Strategy:**
1. Identify highest-level sector containing non-master integrals
2. Run beam search to eliminate all non-masters in that sector
3. Move to next highest sector with non-masters
4. Repeat until only master integrals remain

**Sector processing order for I[2,0,2,0,1,1,0]:**
| Iteration | Sector | Level | Steps | Status |
|-----------|--------|-------|-------|--------|
| 1 | [1,0,1,0,1,1] (53) | 4 | 53 | OK |
| 2 | [0,0,1,0,1,1] (52) | 3 | 17 | OK |
| 3 | [1,0,0,0,1,1] (49) | 3 | 4 | OK |
| 4 | [1,0,1,0,0,1] (37) | 3 | 42 | OK |
| 5 | [1,0,1,0,1,0] (21) | 3 | 46 | OK |
| 6 | [1,0,0,0,0,1] (33) | 2 | 1 | OK |
| 7 | [0,0,1,0,1,0] (20) | 2 | 3 | OK |
| 8 | [1,0,0,0,1,0] (17) | 2 | 1 | OK |
| 9 | [1,0,1,0,0,0] (5) | 2 | 9 | OK |

**Total: 176 steps across 9 sector iterations**

### 4. Sector Filtering

**Filter mode: `subsector`** (strictest)

Only allows IBP actions that produce integrals in strict subsectors of the target. Prevents "lateral" sector pollution that can cause expression blowup.

---

## Key Findings

### Sector 52 Corner Elimination

During sector 52 reduction, the corner integral `I[0,0,1,0,1,1,0]` (which `is_master()` incorrectly labeled as a master) was **correctly eliminated**:

```
Step 15: I[0,0,1,0,1,1,0] coeff 301 -> 0 (elim I[-1,0,2,0,1,1,0] via IBP_5)
```

This confirms that the sector 52 corner is **reducible** to the true paper masters - it's not actually a master integral. AIR kept it as a "master" because it uses a corner basis, while our reduction (like Kira) finds the minimal paper master basis.

### Sector 21 Success

The previous v3 model got stuck on sector 21 (the triangle subsector) at step 95+. Our v5 model successfully reduced it in 46 steps. This was enabled by:
1. Better model guidance from full substitution encoding
2. Improved action scoring
3. Caching optimizations that allowed deeper search

---

## Files Modified/Created

### Core Files
- `models/ibp_env.py` - Added `_raw_eq_cache` and `get_valid_actions_cached()`
- `scripts/eval/beam_search_classifier_v5.py` - Optimized beam search with batching and caching
- `scripts/eval/hierarchical_reduction_v5.py` - Hierarchical sector-by-sector reduction

### Model
- `models/classifier_v5.py` - V5 classifier with `FullSubstitutionEncoder`

---

## Usage

```bash
python -u scripts/eval/hierarchical_reduction_v5.py \
    --integral 2,0,2,0,1,1,0 \
    --beam_width 20 \
    --max_steps 100 \
    --filter_mode subsector \
    --device cuda \
    -v
```

---

## Performance Comparison

**Per-step timing (typical):**
- P1 (get_valid_actions): 0.1-5s (was 10-90s before caching)
- P2 (model inference): 0.03-0.07s (was 0.5s+ before batching)
- P3 (apply actions): 0.1-0.3s

**Total reduction time:** 302 seconds for complete hierarchical reduction

---

## V11-V14: Beam Restart Strategy and Checkpointing

### V11: Beam Restart Innovation

The key insight is that after achieving a weight improvement, the beam often contains many suboptimal states that will never lead to a solution. V11 introduces **beam restart**:

1. Run beam search until max_weight improves
2. When improvement detected, **stop and restart** with only the best state
3. This prunes away dead ends and allows much deeper exploration

**Implementation:** `beam_search()` accepts `stop_on_weight_improvement=True` parameter.

### V14: Checkpoint/Resume

For long-running reductions (e.g., 321322m6 took ~20 hours), crashes are a real risk. V14 adds:

1. **Checkpoint after each sector:** Saves current expression, path, stats
2. **Resume capability:** `--resume checkpoints/dir` continues from saved state
3. **Human-readable summary:** `checkpoint_summary.txt` for monitoring

### V11-V14 Results

| Version | Integral | Weight | Steps | Sectors | Time |
|---------|----------|--------|-------|---------|------|
| V5 | `I[2,0,2,0,1,1,0]` | (6,0) | 176 | 9 | 302s |
| V14 | `I[1,1,1,1,1,1,-3]` | (6,3) | 1,416 | 45 | 353s |
| V12 | `I[3,2,1,3,2,2,-6]` | (13,6) | 46,345 | 62 | 19.7 hr |

The beam restart strategy enables reduction of much higher-weight integrals that V5 could not handle.

### Key Files

- `scripts/eval/beam_search_classifier_v11.py`: Beam search with restart
- `scripts/eval/hierarchical_reduction_v14.py`: Hierarchical reduction with checkpointing
- `scripts/eval/replay_reduction_path.py`: Replay saved reduction paths
- `results/reduction_111111m3_v14.pkl`: Saved 1416-step reduction path

---

## Future Work

1. **Fix `is_master()` function** - Currently returns True for corners in "uncovered" sectors, but these may be reducible. Should only return True for the 16 paper masters.

2. **Larger beam width** - May find shorter reduction paths

3. **Other integrals** - Test on more complex starting integrals in sector 53 and beyond

4. **Training data** - Generate more training examples from successful reductions to improve model

5. **Path optimization** - The saved reduction paths could potentially be shortened by post-processing
