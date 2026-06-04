#!/usr/bin/env python
"""Test classifier determinism on identical inputs.

Test A (PURE DETERMINISM): call model twice on EXACT same input. Are outputs
bit-identical?

Test B (PERMUTATION INVARIANCE): take action list from run A, permute it
to match run B's order. Pass to model. Does the resulting logit-per-action
match what we get from passing run A's original ordering?

Test C (REORDERED INPUT == B's INPUT): take A's actions, reorder to B's,
run model. Compare to running B's actual valid list. Bit-identical?
"""
import argparse
import pickle
import sys
import torch

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, weight,
)
from beam_search_full import prepare_batched_input_v5
from beam_search_utils import get_non_masters
from sailir.classifier import IBPActionClassifier


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def build_iraws(flat, rs, env):
    n_idx = ibp_env.N_INDICES
    iraws = []
    for row in flat.iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        iraws.append((sub_int, op, shift, raw, cached, ub))
    return iraws


def run_model(model, batch_data):
    batch, _ = prepare_batched_input_v5(batch_data, 'cpu')
    with torch.no_grad():
        logits, probs = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
            batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral'],
        )
    return logits, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_a')
    p.add_argument('ckpt_b')
    p.add_argument('topology')
    p.add_argument('--model', required=True)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    torch.set_num_threads(1)

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ca = load(args.ckpt_a)
    cb = load(args.ckpt_b)
    target_sector = tuple(ca['target_sector'])

    by_a = {pk(s): s for s in ca['beam']}
    by_b = {pk(s): s for s in cb['beam']}
    common = sorted(set(by_a) & set(by_b))

    # Pick the first common survivor with a tied target.
    chosen_survivor = None
    chosen_target = None
    for p_ in common:
        sa = by_a[p_]
        nm = get_non_masters(sa['expr'], target_sector)
        if not nm:
            continue
        mw = tuple(sa['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        if tied:
            chosen_survivor = p_
            chosen_target = tied[0]
            break
    if chosen_survivor is None:
        print('No survivor with tied target')
        return 1

    sa = by_a[chosen_survivor]
    sb = by_b[chosen_survivor]
    print(f'Chosen survivor: max_w={sa["max_w"]} nm={sa["n_non_masters"]}')
    print(f'Chosen target: {chosen_target}')

    iraws_a = build_iraws(sa['aux_flat'], sa['resolved_subs'], env)
    iraws_b = build_iraws(sb['aux_flat'], sb['resolved_subs'], env)
    valid_a = enumerate_valid_actions_with_indirect_cache(
        chosen_target, iraws_a, sa['subs'], sa['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    valid_b = enumerate_valid_actions_with_indirect_cache(
        chosen_target, iraws_b, sb['subs'], sb['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    print(f'|valid_A|={len(valid_a)}  |valid_B|={len(valid_b)}')
    print(f'set equal: {set(valid_a) == set(valid_b)}')
    print(f'order equal: {valid_a == valid_b}')

    # === Test A: PURE DETERMINISM ===
    print('\n=== Test A: pure determinism (same input twice) ===')
    bd1 = [(sa['expr'], sa['subs'], valid_a, target_sector, chosen_target)]
    logits1a, probs1a = run_model(model, bd1)
    logits1b, probs1b = run_model(model, bd1)
    same_logits = torch.equal(logits1a, logits1b)
    same_probs = torch.equal(probs1a, probs1b)
    print(f'  Same logits across 2 runs of same input: {same_logits}')
    print(f'  Same probs  across 2 runs of same input: {same_probs}')
    if not same_logits:
        diff = (logits1a - logits1b).abs().max().item()
        print(f'  Max abs diff in logits: {diff!r}')

    # === Test B: A's actions in B's order ===
    print('\n=== Test B: valid_A actions reordered to match valid_B order ===')
    # Build a re-ordered version of valid_a so its positions correspond to valid_b.
    # If sets match, we can permute.
    if set(valid_a) == set(valid_b):
        # Map each action in valid_b to its presence in valid_a; the permuted list = valid_b (since sets match)
        # The act tensor will be exactly the same as if we used valid_b directly.
        perm_a_as_b = valid_b  # same set, just B's order
        bd_perm = [(sa['expr'], sa['subs'], perm_a_as_b, target_sector, chosen_target)]
        logits_perm, probs_perm = run_model(model, bd_perm)
        # Compare to actually running B's input
        bd_b = [(sb['expr'], sb['subs'], valid_b, target_sector, chosen_target)]
        logits_b, probs_b = run_model(model, bd_b)
        same_logits_AB = torch.equal(logits_perm, logits_b)
        print(f'  Same logits when A\'s valid reordered to B\'s order matches B\'s run: {same_logits_AB}')
        if not same_logits_AB:
            diff = (logits_perm - logits_b).abs().max().item()
            print(f'    Max abs diff: {diff!r}')

        # Also: original A vs reordered to B
        same_orig_vs_perm = torch.equal(logits1a, logits_perm)
        print(f'  Same logits A original vs A reordered: {same_orig_vs_perm}')
        if not same_orig_vs_perm:
            diff = (logits1a - logits_perm).abs().max().item()
            print(f'    Max abs diff: {diff!r}')

        # Check multiset equality of logits across positions
        sorted_orig_a = torch.sort(logits1a[0])[0]
        sorted_perm = torch.sort(logits_perm[0])[0]
        same_sorted = torch.equal(sorted_orig_a, sorted_perm)
        print(f'  Sorted logits (multiset) equal between A original and A reordered: {same_sorted}')
        if not same_sorted:
            diff = (sorted_orig_a - sorted_perm).abs().max().item()
            print(f'    Max abs diff in sorted: {diff!r}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
