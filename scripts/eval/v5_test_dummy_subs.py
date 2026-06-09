#!/usr/bin/env python
"""Test what 'dummy subs' configurations preserve model output, by feeding
a step-166 baseline survivor through prepare_batched_input_v5 + classifier
under several subs variants:

A. baseline: subs as-is (control)
B. zero-repls:  sub_keys = real (last 50), sub_repl_ints = 0, sub_repl_coeffs = 0
C. ones-coeffs: sub_keys = real (last 50), sub_repl_ints = real, sub_repl_coeffs = 1
D. random-coeffs: sub_keys = real (last 50), sub_repl_ints = real, sub_repl_coeffs ~ U[1, PRIME-1]
E. empty-subs: no subs at all (mask all zeros)
F. RS-keys, zero-repls: sub_keys = resolved_subs keys (last 50), sub_repl_ints/coeffs zero
G. RS-keys, ones-coeffs: sub_keys = resolved_subs keys, sub_repl_ints = single garbage int, coeffs = 1
H. passenger-stripped: keys/repls from real subs but with values stripped of
   weight < start (i.e. v5's actual feed)

For each variant we run the model on the full valid-action list at step 166
and compare action LOGITS and PROBS against (A).

The "dummy" config we use in v5 is whichever (B-H) matches (A) bit-identically
(or close enough that downstream effect is null).
"""
import argparse
import pickle
import sys
import numpy as np
import torch

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, weight, PRIME,
)
from beam_search_full import prepare_batched_input_v5, MAX_REPLACEMENT_TERMS
from beam_search_utils import get_non_masters
from sailir.classifier import IBPActionClassifier


N_INDICES = None  # set after topology init


def make_subs_variant(state, target, valid_actions, ts, variant, start_w):
    """Return a (expr, subs, valid, ts, target) tuple with subs modified per variant."""
    expr = state['expr']
    subs = state['subs']
    rs = state['resolved_subs']

    if variant == 'A':
        return (expr, subs, valid_actions, ts, target)

    # Build a new subs dict per variant
    if variant == 'E':
        # empty subs
        new_subs = {}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'B':
        # keep subs keys, replace each value with empty dict (so repls all become zero)
        new_subs = {k: {} for k in subs.keys()}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'C':
        # keep subs keys + structure but set all coeffs to 1
        new_subs = {}
        for k, v in subs.items():
            new_subs[k] = {kk: 1 for kk in v.keys()}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'D':
        # random coeffs
        rng = np.random.default_rng(0)
        new_subs = {}
        for k, v in subs.items():
            new_subs[k] = {kk: int(rng.integers(1, PRIME)) for kk in v.keys()}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'F':
        # subs keys = RS keys, values = {}
        new_subs = {k: {} for k in rs.keys()}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'G':
        # subs keys = RS keys, single garbage int with coeff=1
        garbage_int = (0,) * N_INDICES
        new_subs = {k: {garbage_int: 1} for k in rs.keys()}
        return (expr, new_subs, valid_actions, ts, target)

    if variant == 'H':
        # passenger-stripped: keys from subs but values stripped of weight < start
        def is_active(integral):
            w = weight(integral)
            return (w[0], w[1]) >= (start_w[0], start_w[1])
        new_subs = {}
        for k, v in subs.items():
            new_v = {kk: cc for kk, cc in v.items() if is_active(kk)}
            new_subs[k] = new_v
        return (expr, new_subs, valid_actions, ts, target)

    raise ValueError(f'unknown variant {variant}')


def run_model(model, batch_data, device='cpu'):
    b, _ = prepare_batched_input_v5(batch_data, device)
    with torch.no_grad():
        logits, probs = model(
            b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
            b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
            b['sub_repl_mask'], b['sub_mask'],
            b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
            b['sector_mask'], b['target_integral'],
        )
    return logits[0].numpy(), probs[0].numpy()


def main():
    global N_INDICES
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('topology')
    p.add_argument('--model', required=True)
    p.add_argument('--integral', default='-1,2,1,0,1,2,1,1,-3,0,0',
                   help='Starting integral (defines active threshold)')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    torch.set_num_threads(1)

    N_INDICES = ibp_env.N_INDICES
    start_int = tuple(int(x) for x in args.integral.split(','))
    start_w = weight(start_int)
    print(f'Starting integral: {start_int}  weight={start_w}')
    print(f'Active threshold = (w1,w2) >= ({start_w[0]}, {start_w[1]})')

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    with open(args.ckpt, 'rb') as f:
        c = pickle.load(f)
    ts = tuple(c['target_sector'])
    s = c['beam'][0]
    nm = get_non_masters(s['expr'], ts)
    mw = tuple(s['max_w'])
    tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
    target = tied[0]
    print(f'Survivor: max_w={mw} target={target}  |subs|={len(s["subs"])} '
          f'|RS|={len(s["resolved_subs"])}')

    # Build iraws and enumerate valid
    rs = s['resolved_subs']
    iraws = []
    for row in s['aux_flat'].iraws_meta:
        sub_int = tuple(int(x) for x in row[:N_INDICES])
        op = int(row[N_INDICES])
        shift = tuple(int(x) for x in row[N_INDICES + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(N_INDICES))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        iraws.append((sub_int, op, shift, raw, cached, ub))

    valid = enumerate_valid_actions_with_indirect_cache(
        target, iraws, s['subs'], rs,
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    print(f'|valid|={len(valid)}')

    # Reference: variant A
    bd_A = [make_subs_variant(s, target, valid, ts, 'A', start_w)]
    logits_A, probs_A = run_model(model, bd_A)
    nv = (logits_A != 0).sum()  # rough; mask is what really matters
    print(f'\nReference (A): logits shape={logits_A.shape}  '
          f'(first 5 logits = {logits_A[:5].round(4).tolist()})')

    print(f'\n{"variant":<8} {"|logit Δ|max":>14} {"|prob Δ|max":>14} '
          f'{"prob top1 match":>16}')
    top1_A = int(np.argmax(probs_A))
    for variant in ('B', 'C', 'D', 'E', 'F', 'G', 'H'):
        bd = [make_subs_variant(s, target, valid, ts, variant, start_w)]
        logits_X, probs_X = run_model(model, bd)
        # Compare across the masked (used) region: model fills 0 outside valid
        diff_l = float(np.max(np.abs(logits_X - logits_A)))
        diff_p = float(np.max(np.abs(probs_X - probs_A)))
        top1_X = int(np.argmax(probs_X))
        same_top1 = (top1_X == top1_A)
        print(f'{variant:<8} {diff_l:>14.3e} {diff_p:>14.3e}  {str(same_top1):>16}')


if __name__ == '__main__':
    sys.exit(main())
