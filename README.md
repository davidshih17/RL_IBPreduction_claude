# Neural Network Guided IBP Reduction

This repository contains code for training a neural network to guide Integration-by-Parts (IBP) reduction of Feynman loop integrals.

## Key Result

A transformer-based classifier trained on scrambled reduction paths successfully discovers a complete IBP reduction for the integral `I[2,0,2,0,1,1,0]` using beam search, reducing it to master integrals in 55 steps.

## Background

IBP reduction is a critical technique in particle physics for simplifying loop integrals. Given an integral, the goal is to express it as a linear combination of simpler "master" integrals using IBP identities.

**Master integrals for this problem:**
- `I[1, 0, 1, 0, 1, 1, 0]`
- `I[1, -1, 1, 0, 1, 1, 0]`

## Method

### Training Data Generation
- Start from master integrals and "scramble" by applying random IBP operations in reverse
- This generates (state, action) pairs where the action leads toward reduction
- Filter out actions that introduce "higher sector" integrals (positions 1, 3, or 6 become positive)
- Use a small prime (p=1009) for modular arithmetic to keep coefficients manageable

### Model Architecture
- **Classifier v3**: Transformer-based encoder for expressions
- Input: current expression (list of integrals with coefficients), substitution history, valid actions
- Output: probability distribution over valid actions
- Trained with cross-entropy loss to predict the "correct" reverse-scramble action

### Inference
- Beam search with the trained model
- At each step, eliminate the highest-weight non-master integral
- Model scores all valid actions; explore top-k
- Prioritize states with lower maximum weight

## Repository Structure

```
├── models/
│   ├── ibp_env.py           # IBP environment, action enumeration
│   ├── classifier_v1.py     # Base classifier components
│   ├── classifier_v2.py     # Extended encoders
│   └── classifier_v3.py     # Transformer classifier (main model)
├── scripts/
│   ├── data_gen/
│   │   ├── generate_classifier_data_small_prime.py
│   │   ├── IBP                # IBP equation templates
│   │   └── LI                 # Lorentz identity templates
│   ├── training/
│   │   └── train_classifier_v3.py
│   └── eval/
│       └── beam_search_classifier_v3.py
├── reference/
│   └── reverse_reduction_reordered.log  # AIR's 32-step reference
├── data/
│   └── classifier_training_data_p1009_filtered.jsonl.gz
├── checkpoints/
│   └── best_model.pt        # Trained model (95.77% val accuracy)
└── logs/
    ├── train_classifier_v3_filtered.log
    └── beam_search_v3.log   # Successful 55-step reduction
```

## Usage

### Generate Training Data
```bash
python -u scripts/data_gen/generate_classifier_data_small_prime.py \
    --num_scrambles 2000 --max_steps 20 \
    --output data/classifier_training_data_p1009_filtered.jsonl
```

### Train Model
```bash
python -u scripts/training/train_classifier_v3.py \
    --data data/classifier_training_data_p1009_filtered.jsonl \
    --epochs 30 --batch_size 32 --prime 1009
```

### Run Beam Search Reduction
```bash
python -u scripts/eval/beam_search_classifier_v3.py \
    --checkpoint checkpoints/best_model.pt \
    --beam_width 10 --max_steps 100
```

## Results

### Training
- 24,589 training samples from 2000 scrambles
- Best validation accuracy: 95.77%
- See `logs/train_classifier_v3_filtered.log`

### Beam Search Reduction
- Starting integral: `I[2,0,2,0,1,1,0]` (weight 6,0)
- Successfully reduced to masters in 55 steps
- Weight progression: (6,0) → (5,0) → (4,1) → complete
- See `logs/beam_search_v3.log` for full path

### Comparison to AIR
- AIR (traditional method): 32 steps
- Neural-guided beam search: 55 steps
- The neural approach finds a valid (if longer) reduction path entirely through learned heuristics

## Key Insights

1. **Training on scrambled paths works**: Even without access to optimal reduction paths, training on random scrambles teaches useful structure

2. **Higher sector filtering is critical**: Actions that introduce integrals with positive indices at positions 1, 3, or 6 lead to explosion of complexity

3. **Weight-based target selection**: Always eliminating the highest-weight non-master integral provides a good greedy heuristic

4. **Beam search enables discovery**: The model alone may not always pick the optimal action, but beam search explores enough to find valid paths

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for faster training)

## License

MIT
