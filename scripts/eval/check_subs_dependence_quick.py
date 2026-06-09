"""Run delta beam search a few steps on (8,4), capture the model input at
a chosen step, then re-run the model with sub_mask zeroed out. Compare
logits + top-k action selection to test whether the model is empirically
subs-insensitive.

Usage:
    python scripts/eval/check_subs_dependence_quick.py \
        --topology topology_input/pentagonbox \
        --model-checkpoint checkpoints/pentagonbox_10x_loop_100/best_model.pt \
        --capture-step 50 --max-steps 50
"""
import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, set_paper_masters_only, weight,
)
from sailir.classifier import IBPActionClassifier
import sailir.delta_beam_search as dbs
from beam_search_utils import get_sector_mask


def load_model(checkpoint_path, topology, device='cpu'):
    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--integral', type=str, default='-1,2,1,0,1,2,1,1,-3,0,0')
    parser.add_argument('--capture-step', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=40)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    model = load_model(args.model_checkpoint, topology, args.device)

    integral = tuple(int(x) for x in args.integral.split(','))
    target_sector = tuple(get_sector_mask(integral))
    start_expr = {integral: 1}

    # Monkey-patch the model.forward to capture inputs at the chosen step.
    captured = {'step': 0, 'inputs': None, 'logits': None, 'probs': None}
    orig_forward = model.forward

    def hook_forward(expr_integrals, expr_coeffs, expr_mask,
                     sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                     action_ibp_ops, action_deltas, action_mask,
                     sector_mask, target_integral):
        result = orig_forward(
            expr_integrals, expr_coeffs, expr_mask,
            sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
            action_ibp_ops, action_deltas, action_mask,
            sector_mask, target_integral,
        )
        if captured['step'] == args.capture_step and captured['inputs'] is None:
            captured['inputs'] = {
                'expr_integrals': expr_integrals.clone(),
                'expr_coeffs': expr_coeffs.clone(),
                'expr_mask': expr_mask.clone(),
                'sub_keys': sub_keys.clone(),
                'sub_repl_ints': sub_repl_ints.clone(),
                'sub_repl_coeffs': sub_repl_coeffs.clone(),
                'sub_repl_mask': sub_repl_mask.clone(),
                'sub_mask': sub_mask.clone(),
                'action_ibp_ops': action_ibp_ops.clone(),
                'action_deltas': action_deltas.clone(),
                'action_mask': action_mask.clone(),
                'sector_mask': sector_mask.clone(),
                'target_integral': target_integral.clone(),
            }
            captured['logits'] = result[0].clone()
            captured['probs'] = result[1].clone()
            print(f'  [hook] captured step {args.capture_step}: '
                  f'batch={expr_integrals.shape[0]} '
                  f'|sub_mask True|/batch={sub_mask.sum(dim=1).float().mean():.1f}',
                  flush=True)
        captured['step'] += 1
        return result

    model.forward = hook_forward

    print(f'Running beam_search_delta for {args.max_steps} steps to capture step {args.capture_step}...')
    t0 = time.time()
    solution, beam, best_w = dbs.beam_search_delta(
        env, model, start_expr, target_sector,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        prime=args.prime,
        verbose=False,
        stop_on_weight_improvement=False,
        device=args.device,
    )
    print(f'beam search ran {time.time()-t0:.1f}s\n')

    if captured['inputs'] is None:
        print(f'ERROR: never captured step {args.capture_step}. Try lower.')
        return

    # Restore original forward
    model.forward = orig_forward

    # Run the captured batch unchanged
    with torch.no_grad():
        logits_real, probs_real = model(**captured['inputs'])

    # Test variant 1: zeroed sub_mask (no subs)
    inputs_empty = dict(captured['inputs'])
    inputs_empty['sub_mask'] = torch.zeros_like(captured['inputs']['sub_mask'])
    with torch.no_grad():
        logits_empty, probs_empty = model(**inputs_empty)

    # Test variant 2: shift subs (state i gets state (i+1 mod batch)'s subs).
    # This keeps inputs IN-DISTRIBUTION (real subs from a real state) but
    # mismatched to the state's actual expr/RS — a sanity check for whether
    # the model cares about the specific sub history vs just having
    # realistic-shaped subs present.
    n_batch = captured['inputs']['expr_integrals'].shape[0]
    perm = torch.cat([torch.arange(1, n_batch), torch.tensor([0])])
    inputs_swapped = dict(captured['inputs'])
    inputs_swapped['sub_keys'] = captured['inputs']['sub_keys'][perm].clone()
    inputs_swapped['sub_repl_ints'] = captured['inputs']['sub_repl_ints'][perm].clone()
    inputs_swapped['sub_repl_coeffs'] = captured['inputs']['sub_repl_coeffs'][perm].clone()
    inputs_swapped['sub_repl_mask'] = captured['inputs']['sub_repl_mask'][perm].clone()
    inputs_swapped['sub_mask'] = captured['inputs']['sub_mask'][perm].clone()
    with torch.no_grad():
        logits_swapped, probs_swapped = model(**inputs_swapped)

    # Compare.
    n_batch = logits_real.shape[0]
    print(f'\n=== model output comparison at captured step {args.capture_step} ===')
    print(f'batch_size = {n_batch}\n')

    import statistics
    nva = captured['inputs']['action_mask'].sum(dim=1).long()

    def stats(logits_other, probs_other, label):
        max_logit_diffs = []
        prob_diffs = []
        top1_same = 0
        top5_overlap = []
        for i in range(n_batch):
            n = int(nva[i].item())
            if n == 0:
                continue
            d = (logits_real[i, :n] - logits_other[i, :n]).abs().max().item()
            max_logit_diffs.append(d)
            pd = (probs_real[i, :n] - probs_other[i, :n]).abs().max().item()
            prob_diffs.append(pd)
            t1_a = int(torch.argmax(logits_real[i, :n]).item())
            t1_b = int(torch.argmax(logits_other[i, :n]).item())
            if t1_a == t1_b:
                top1_same += 1
            k = min(5, n)
            tk_a = set(torch.topk(logits_real[i, :n], k).indices.tolist())
            tk_b = set(torch.topk(logits_other[i, :n], k).indices.tolist())
            top5_overlap.append(len(tk_a & tk_b) / k)
        print(f'\n--- {label} ---')
        print(f'  max-abs logit diff per task: '
              f'mean={statistics.mean(max_logit_diffs):.4f} '
              f'median={statistics.median(max_logit_diffs):.4f} '
              f'max={max(max_logit_diffs):.4f}')
        print(f'  max-abs prob diff per task:  '
              f'mean={statistics.mean(prob_diffs):.4f} '
              f'median={statistics.median(prob_diffs):.4f} '
              f'max={max(prob_diffs):.4f}')
        print(f'  top-1 action same:    {top1_same}/{len(max_logit_diffs)} tasks '
              f'({100*top1_same/len(max_logit_diffs):.1f}%)')
        print(f'  top-5 overlap (avg):  {statistics.mean(top5_overlap):.2f}/5 '
              f'({100*statistics.mean(top5_overlap):.1f}%)')

    stats(logits_empty, probs_empty, 'zeroed sub_mask (out-of-distribution test)')
    stats(logits_swapped, probs_swapped, 'subs from neighbor state (in-distribution test)')

    # Test variant 2.5: RANDOM subs. Completely random integer values within
    # the observed range of the captured subs. Tests whether the model relies
    # on subs being "structurally meaningful" or whether it accepts any noise.
    real_keys = captured['inputs']['sub_keys']
    real_ints = captured['inputs']['sub_repl_ints']
    real_coeffs = captured['inputs']['sub_repl_coeffs']

    key_lo, key_hi = int(real_keys.min().item()), int(real_keys.max().item())
    int_lo, int_hi = int(real_ints.min().item()), int(real_ints.max().item())
    coeff_max = int(real_coeffs.max().item())

    torch.manual_seed(0)
    inputs_random = dict(captured['inputs'])
    inputs_random['sub_keys'] = torch.randint(key_lo, key_hi + 1, real_keys.shape).to(real_keys.dtype)
    inputs_random['sub_repl_ints'] = torch.randint(int_lo, int_hi + 1, real_ints.shape).to(real_ints.dtype)
    inputs_random['sub_repl_coeffs'] = torch.randint(1, max(coeff_max, 2) + 1, real_coeffs.shape).to(real_coeffs.dtype)
    # keep sub_mask, sub_repl_mask as the originals (same shape semantics)
    print(f'\n  random-subs ranges: '
          f'keys in [{key_lo},{key_hi}] '
          f'repl_ints in [{int_lo},{int_hi}] '
          f'coeffs in [1,{coeff_max}]')
    with torch.no_grad():
        logits_random, probs_random = model(**inputs_random)
    stats(logits_random, probs_random,
          'COMPLETELY RANDOM sub_keys + sub_repl_ints + sub_repl_coeffs '
          '(masks unchanged)')

    # Sanity: are the original subs actually different across batch?
    sub_keys = captured['inputs']['sub_keys']
    n_distinct = 0
    for i in range(1, n_batch):
        if not torch.equal(sub_keys[0], sub_keys[i]):
            n_distinct += 1
    print(f'\nsanity: sub_keys[0] vs sub_keys[i] differ in {n_distinct}/{n_batch-1} batch positions')
    # Print first 5 entries of state 0 and state 50 to eyeball
    print(f'  state 0 sub_keys[:5]:  {sub_keys[0, :5].tolist()}')
    if n_batch > 50:
        print(f'  state 50 sub_keys[:5]: {sub_keys[50, :5].tolist()}')

    # Test variant 3: REVERSE the subs order in each state. This tests
    # order sensitivity. Model has positional encoding so this should differ
    # if model uses positions.
    inputs_rev = dict(captured['inputs'])
    inputs_rev['sub_keys'] = captured['inputs']['sub_keys'].flip(dims=[1]).clone()
    inputs_rev['sub_repl_ints'] = captured['inputs']['sub_repl_ints'].flip(dims=[1]).clone()
    inputs_rev['sub_repl_coeffs'] = captured['inputs']['sub_repl_coeffs'].flip(dims=[1]).clone()
    inputs_rev['sub_repl_mask'] = captured['inputs']['sub_repl_mask'].flip(dims=[1]).clone()
    inputs_rev['sub_mask'] = captured['inputs']['sub_mask'].flip(dims=[1]).clone()
    with torch.no_grad():
        logits_rev, probs_rev = model(**inputs_rev)
    stats(logits_rev, probs_rev, 'subs order REVERSED within each state (order-sensitivity test)')


if __name__ == '__main__':
    main()
