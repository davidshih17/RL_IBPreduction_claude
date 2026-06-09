#!/usr/bin/env python
"""For a step-166 survivor whose (expr, subs, RS) matches between baseline
and v4, enumerate valid in each (using each run's iraws), take first 900,
run model on each, and:
  - Verify Phase 1a action LOGITS are identical (sanity: same action features)
  - Compare Phase 1a action PROBABILITIES (different Z from different
    Phase 1b filler)
  - Compare softmax Z over first 900 between baseline and v4
If probabilities differ for the SAME Phase 1a action, that proves
the score CAN drift through softmax-denominator drift.
"""
import argparse
import pickle
import sys
import torch
import numpy as np

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


def pk(s):
    return tuple(tuple(a) for a in s['path'])


def build_iraws(sp, env):
    n_idx = ibp_env.N_INDICES
    rs = sp['resolved_subs']
    out = []
    for row in sp['aux_flat'].iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        cached = apply_resolved_subs(raw, rs)
        ub = cached_union_bitmask(cached)
        out.append((sub_int, op, shift, raw, cached, ub))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-ckpt', required=True)
    p.add_argument('--v4-ckpt', required=True)
    p.add_argument('--topology', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--max-actions', type=int, default=900)
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
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    with open(args.baseline_ckpt, 'rb') as f:
        bp = pickle.load(f)
    with open(args.v4_ckpt, 'rb') as f:
        vp = pickle.load(f)
    ts = tuple(bp['target_sector'])

    # Match by path; take first survivor that matches in (expr, subs, RS)
    by_v = {pk(s): s for s in vp['beam']}
    survivor_b = None
    survivor_v = None
    for sb in bp['beam']:
        p_ = pk(sb)
        if p_ not in by_v:
            continue
        sv = by_v[p_]
        if (sb['expr'] == sv['expr'] and sb['subs'] == sv['subs']
                and sb['resolved_subs'] == sv['resolved_subs']):
            survivor_b = sb
            survivor_v = sv
            break
    assert survivor_b is not None, 'no matching survivor'
    print(f'Found matching survivor. baseline.score={survivor_b["score"]:.6f} '
          f'v4.score={survivor_v["score"]:.6f} '
          f'drift={survivor_v["score"]-survivor_b["score"]:+.6f}')

    nm = get_non_masters(survivor_b['expr'], ts)
    mw = tuple(survivor_b['max_w'])
    tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
    target = tied[0]
    print(f'Using target={target}  mw={mw}  |tied|={len(tied)}  |nm|={len(nm)}')

    ia = build_iraws(survivor_b, env)
    iv = build_iraws(survivor_v, env)
    print(f'baseline iraws_meta len={len(survivor_b["aux_flat"].iraws_meta)}  '
          f'v4 iraws_meta len={len(survivor_v["aux_flat"].iraws_meta)}')

    va = enumerate_valid_actions_with_indirect_cache(
        target, ia, survivor_b['subs'], survivor_b['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    vv = enumerate_valid_actions_with_indirect_cache(
        target, iv, survivor_v['subs'], survivor_v['resolved_subs'],
        env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
    )
    print(f'|valid baseline|={len(va)}  |valid v4|={len(vv)}  set-equal? {set(va)==set(vv)}')

    # First N in each
    N = args.max_actions
    a_first = va[:N]
    v_first = vv[:N]
    same_order = (a_first == v_first)
    print(f'first {N} same order? {same_order}')
    if not same_order:
        # how many shared in first N
        sa = set(a_first); sv = set(v_first)
        shared = sa & sv
        print(f'  shared in both first-N: {len(shared)}  '
              f'baseline-only: {len(sa-sv)}  v4-only: {len(sv-sa)}')

    # Run model on baseline-first-N and on v4-first-N
    def runmodel(state, valid_subset):
        bd = [(state['expr'], state['subs'], valid_subset, ts, target)]
        b, _ = prepare_batched_input_v5(bd, 'cpu')
        with torch.no_grad():
            logits, probs = model(
                b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                b['sub_repl_mask'], b['sub_mask'],
                b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                b['sector_mask'], b['target_integral'],
            )
        nv = min(len(valid_subset), probs.shape[1])
        return logits[0, :nv].numpy(), probs[0, :nv].numpy()

    la, pa = runmodel(survivor_b, a_first)
    lv, pv = runmodel(survivor_v, v_first)

    # For each action present in both first-N windows, compare logit and prob
    pos_a = {act: i for i, act in enumerate(a_first)}
    pos_v = {act: i for i, act in enumerate(v_first)}
    shared = set(pos_a) & set(pos_v)
    n_logit_match = 0
    n_logit_close = 0
    n_logit_diff = 0
    n_prob_match = 0
    n_prob_close = 0
    n_prob_diff = 0
    max_logit_diff = 0
    max_prob_diff = 0
    for act in shared:
        ia2 = pos_a[act]
        iv2 = pos_v[act]
        ldiff = abs(la[ia2] - lv[iv2])
        pdiff = abs(pa[ia2] - pv[iv2])
        max_logit_diff = max(max_logit_diff, ldiff)
        max_prob_diff = max(max_prob_diff, pdiff)
        if la[ia2] == lv[iv2]: n_logit_match += 1
        elif ldiff < 1e-6: n_logit_close += 1
        else: n_logit_diff += 1
        if pa[ia2] == pv[iv2]: n_prob_match += 1
        elif pdiff < 1e-6: n_prob_close += 1
        else: n_prob_diff += 1

    print(f'\nShared actions (in both first-{N}): {len(shared)}')
    print(f'Logits: bit-equal={n_logit_match}  <1e-6={n_logit_close}  '
          f'>=1e-6={n_logit_diff}  max|Δ|={max_logit_diff:.3e}')
    print(f'Probs:  bit-equal={n_prob_match}  <1e-6={n_prob_close}  '
          f'>=1e-6={n_prob_diff}  max|Δ|={max_prob_diff:.3e}')

    # Softmax Z = sum(exp(logits)) over the N actions
    Z_a = float(np.exp(la).sum())
    Z_v = float(np.exp(lv).sum())
    print(f'\nSoftmax Z over first {N}:')
    print(f'  baseline Z = {Z_a:.10e}')
    print(f'  v4       Z = {Z_v:.10e}')
    print(f'  ratio v4/base = {Z_v/Z_a:.10f}  log(ratio) = {np.log(Z_v/Z_a):+.6e}')


if __name__ == '__main__':
    sys.exit(main())
