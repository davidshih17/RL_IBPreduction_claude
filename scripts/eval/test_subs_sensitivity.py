#!/usr/bin/env python3
"""Test whether the model's output is sensitive to the subs list.

Take one (expr, target) configuration from a real beam state, run the model
with several different subs contexts:
    - full subs (last 50)
    - first 5 of the 50
    - last 5 of the 50
    - middle 5 of the 50
    - empty subs
    - shuffled subs (random permutation)
For each, compute logit vector over valid actions. Compare:
    - argmax (top-1 action chosen)
    - top-5 actions
    - L1 distance and KL divergence between probability distributions

If the model effectively ignores subs, all variants give ~identical outputs.
"""
import os
import sys
import pickle
import random
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
import numpy as np

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    filter_subs_to_exact_sector, filter_resolved_subs_to_exact_sector,
)
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask
from beam_search import prepare_batched_input_v5

CHECKPOINT = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v3/work/results/straggler_19642_-1_2_1_0_1_2_1_1_-3_0_0.pkl.checkpoint'
TOPOLOGY = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'
MODEL_PATH = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt'

t = Topology.from_dir(TOPOLOGY)
ibp_env.init_from_topology(t)
set_prime(1009)
env = IBPEnvironment()

# Load model
print('Loading model...')
model = IBPActionClassifier(
    n_indices=t.n_indices,
    n_denominators=t.n_denominators,
    n_ibp_ops=t.n_actions,
)
ck_model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(ck_model['model_state_dict'])
model.eval()

print('Loading state checkpoint...')
with open(CHECKPOINT, 'rb') as f:
    ck = pickle.load(f)
s0 = ck['beam'][0]
INTEGRAL = (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0)
target_sector = tuple(get_sector_mask(INTEGRAL))
fsubs = filter_subs_to_exact_sector(s0.subs, target_sector)
fresolved = filter_resolved_subs_to_exact_sector(s0.resolved_subs, target_sector)
print(f'state[0]: |expr|={len(s0.expr)}, |subs|={len(s0.subs)}, |fresolved|={len(fresolved)}')

# Pick target = highest-weight non-master in target sector
from sailir.ibp_env import weight, is_master
non_masters = [(weight(k)[:2], k) for k, v in s0.expr.items() if v != 0 and not is_master(k)]
if not non_masters:
    print('No non-masters; bailing.'); sys.exit(1)
non_masters.sort(reverse=True)
target = non_masters[0][1]
print(f'target = {list(target)}, weight={weight(target)[:2]}')

# Get valid actions for this target with the FULL filtered_subs/resolved
indirect_cache = env.compute_indirect_cache(fsubs, fresolved)
valid_actions = env.get_valid_actions_with_cache(target, indirect_cache, fsubs, fresolved, filter_mode='subsector')
print(f'|valid_actions| = {len(valid_actions)}')


def predict(subs_dict, label):
    """Run model with the given subs context (already filtered) for (expr, target, valid_actions)."""
    batch_data = [(s0.expr, subs_dict, valid_actions, target_sector, target)]
    batch, n_valid_per = prepare_batched_input_v5(batch_data, device='cpu')
    with torch.no_grad():
        logits, _ = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
            batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral'],
        )
    nv = n_valid_per[0]
    lg = logits[0, :nv]
    probs = torch.softmax(lg, dim=-1)
    return lg.cpu().numpy(), probs.cpu().numpy()


# Convert fsubs items to a list so we can slice (insertion-order preserved)
all_items = list(fsubs.items())
n = len(all_items)
print(f'Total filtered subs: {n}\n')

# Last 50 is what the model normally sees
last50_items = all_items[-50:]


def make_subs(items_list):
    return dict(items_list)


configs = [
    ('full last 50 (baseline)', last50_items),
    ('only first 5 of last 50', last50_items[:5]),
    ('only last 5 of last 50', last50_items[-5:]),
    ('only middle 5 of last 50', last50_items[22:27]),
    ('empty subs', []),
]

# Add shuffled-last-50
import copy
shuf = copy.deepcopy(last50_items)
random.Random(42).shuffle(shuf)
configs.append(('shuffled last 50', shuf))

results = {}
for label, items in configs:
    subs = make_subs(items)
    lg, pr = predict(subs, label)
    results[label] = (lg, pr)
    top5 = np.argsort(-lg)[:5]
    print(f'{label:30s}  argmax={int(top5[0]):4d}  top5={list(top5)}  '
          f'max_p={float(pr.max()):.4f}')

print()
baseline_lg, baseline_pr = results['full last 50 (baseline)']
print(f'{"comparison vs baseline":30s} {"L1 logits":>10s} {"L_inf logits":>12s} {"KL prob":>10s} {"argmax agree":>14s}  '
      f'top5 jaccard')
print('-' * 100)
for label, (lg, pr) in results.items():
    if label == 'full last 50 (baseline)':
        continue
    l1 = np.abs(lg - baseline_lg).sum()
    linf = np.abs(lg - baseline_lg).max()
    kl = (baseline_pr * (np.log(baseline_pr + 1e-12) - np.log(pr + 1e-12))).sum()
    agree = (np.argmax(lg) == np.argmax(baseline_lg))
    top5_b = set(np.argsort(-baseline_lg)[:5].tolist())
    top5_o = set(np.argsort(-lg)[:5].tolist())
    j = len(top5_b & top5_o) / len(top5_b | top5_o)
    print(f'{label:30s} {l1:10.4f} {linf:12.4f} {kl:10.6f} {str(agree):>14s}  {j:.2f}')
