#!/usr/bin/env python
"""Replay a v5 path and classify each chosen action by its iraws anchor.

For each chosen action (target, op, delta) along the path:
  - Check Phase 1a (anchor = target): delta == -shift for some topology shift,
    raw at seed=target-shift has target with nonzero coeff.
  - Else Phase 1b (anchor = some past sub_int): find sub_int such that
    sub_int = target + delta + shift for some topology shift, and (sub_int, op,
    shift) is in the iraws (sub_int is in RS at this step + raw has sub_int
    with nonzero coeff).

For Phase 1b anchors, record the anchor's "age" — how many steps ago that
sub_int was added to RS.

By default analyzes beam[0]'s path. Pass --beam-idx N to pick another.
"""
import argparse
import pickle
import sys
from collections import Counter

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, solve_ibp_for, weight,
)
from beam_search_v5 import (
    apply_substitution_v5, add_sub_to_resolved_v5, State_v5,
)
from beam_search_utils import get_non_masters, get_sector_mask


def classify_action(target, chosen_op, chosen_delta, rs_keys, env, n_idx):
    """Identify how the chosen action could be generated:
      - 'p1a': Phase 1a (anchor = target)
      - 'p1b@k': Phase 1b anchored on RS[k] (anchor is k-th sub_int added)
      - 'p1b@multi:[k1,k2,...]': matched by multiple anchors
    Returns (classification, anchor_age) where anchor_age = how many steps
    back the anchor was added (relative to current RS size), or None for p1a.
    """
    chosen_delta = tuple(chosen_delta)
    # Phase 1a check: delta == -shift for some topology shift with op,
    # and raw at seed=target-shift has target nonzero.
    p1a = False
    for ibp_op, shift_list in env.shifts.items():
        if ibp_op != chosen_op:
            continue
        for shift in shift_list:
            neg_shift = tuple(-s for s in shift)
            if neg_shift != chosen_delta:
                continue
            seed = tuple(target[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(chosen_op, seed)
            if target in raw and raw[target] != 0:
                p1a = True
                break
        if p1a:
            break

    # Phase 1b candidates: find sub_int = target + delta + shift such that
    # sub_int ∈ RS and raw_at_(sub_int - shift) has sub_int with nonzero coeff.
    p1b_anchors = []
    rs_key_set = set(rs_keys)
    rs_key_index = {k: i for i, k in enumerate(rs_keys)}
    n_rs = len(rs_keys)
    for ibp_op, shift_list in env.shifts.items():
        if ibp_op != chosen_op:
            continue
        for shift in shift_list:
            sub_int = tuple(target[i] + chosen_delta[i] + shift[i]
                            for i in range(n_idx))
            if sub_int not in rs_key_set:
                continue
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(chosen_op, seed)
            if sub_int in raw and raw[sub_int] != 0:
                p1b_anchors.append(sub_int)

    # Classify
    if not p1b_anchors:
        if p1a:
            return 'p1a', None
        return 'unknown', None
    # Choose the OLDEST anchor (most negative age, i.e. earliest in RS)
    # baseline iraws would use the first sub_int (oldest in insertion order)
    # that produces this action.
    ages = [n_rs - 1 - rs_key_index[a] for a in p1b_anchors]
    oldest_age = max(ages)  # largest age = oldest anchor
    if p1a:
        # Action is reachable via both Phase 1a AND Phase 1b. The enumerate
        # code dedups (op, delta) so it depends which phase emitted first.
        # In the enumerator, Phase 1a runs first. So Phase 1a wins.
        return 'p1a_or_p1b', oldest_age
    return f'p1b', oldest_age


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True,
                   help='v5 tabu result.pkl (has beam with paths)')
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--beam-idx', type=int, default=0,
                   help='Which beam[i]\'s path to analyze (default 0)')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    start_int = tuple(int(x.strip("'\"")) for x in args.integral.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))

    with open(args.result, 'rb') as f:
        d = pickle.load(f)
    # Use best_state if it has a path, else beam[idx]
    if d.get('best_state') and d['best_state'].get('path'):
        path = d['best_state']['path']
        print(f'Analyzing best_state path (len {len(path)})')
    else:
        path = d['beam'][args.beam_idx]['path']
        print(f'Analyzing beam[{args.beam_idx}] path (len {len(path)})')

    n_idx = ibp_env.N_INDICES

    # Replay state step-by-step. At each step, classify the chosen action
    # given the CURRENT RS (before this step's action is applied).
    state = State_v5(
        expr={start_int: 1},
        resolved_subs={},
        sub_accum={},
        score=0.0,
        path=[],
        n_non_masters=1,
        max_w12=start_w12,
        total_w12=start_w12,
    )

    counts = Counter()
    ages = []
    age_by_step = []  # list of (step, age) for plotting later if needed
    p1a_count = 0
    p1b_count = 0
    p1a_or_p1b_count = 0
    unknown_count = 0

    for step, (target, ibp_op, delta) in enumerate(path):
        rs_keys = list(state.resolved_subs.keys())
        n_rs = len(rs_keys)
        kind, age = classify_action(target, ibp_op, delta, rs_keys, env, n_idx)
        counts[kind] += 1
        if kind == 'p1a':
            p1a_count += 1
        elif kind == 'p1b':
            p1b_count += 1
            ages.append(age)
            age_by_step.append((step + 1, age, n_rs))
        elif kind == 'p1a_or_p1b':
            p1a_or_p1b_count += 1
        else:
            unknown_count += 1

        # Advance state (apply the action like v5 does).
        seed = tuple(target[i] + delta[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        cached = apply_resolved_subs(raw, state.resolved_subs)
        if target not in cached or cached[target] == 0:
            print(f'  step {step+1}: REPLAY FAILED — target not in cached')
            break
        sol = solve_ibp_for(cached, target)
        new_expr, new_sub_accum = apply_substitution_v5(
            state.expr, state.sub_accum, target, sol, target_sector, start_w12,
        )
        new_rs = add_sub_to_resolved_v5(
            state.resolved_subs, target, sol, start_w12,
        )
        nm = get_non_masters(new_expr, target_sector)
        if nm:
            wl = [(weight(k)[0], weight(k)[1]) for k in nm]
            mw = max(wl)
            tw = (sum(w[0] for w in wl), sum(w[1] for w in wl))
        else:
            mw, tw = (0, 0), (0, 0)
        state = State_v5(
            expr=new_expr, resolved_subs=new_rs, sub_accum=new_sub_accum,
            score=0.0, path=[], n_non_masters=len(nm),
            max_w12=mw, total_w12=tw,
        )

    print(f'\n=== Action classification across {len(path)} steps ===')
    print(f'  Phase 1a only:      {p1a_count}  ({p1a_count/len(path)*100:.1f}%)')
    print(f'  Phase 1b only:      {p1b_count}  ({p1b_count/len(path)*100:.1f}%)')
    print(f'  Phase 1a OR 1b:     {p1a_or_p1b_count}  ({p1a_or_p1b_count/len(path)*100:.1f}%)')
    print(f'  Unknown (neither):  {unknown_count}  ({unknown_count/len(path)*100:.1f}%)')

    if ages:
        import statistics
        print(f'\n=== Phase 1b anchor age (steps since anchor added to RS) ===')
        print(f'  n samples:       {len(ages)}')
        print(f'  min:             {min(ages)}')
        print(f'  max:             {max(ages)}')
        print(f'  median:          {statistics.median(ages):.0f}')
        print(f'  mean:            {statistics.mean(ages):.1f}')
        # Distribution buckets
        buckets = [(0,4), (5,9), (10,19), (20,49), (50,99), (100,199), (200,500)]
        print(f'\n  Distribution:')
        for lo, hi in buckets:
            n = sum(1 for a in ages if lo <= a <= hi)
            print(f'    age [{lo:>3}-{hi:>3}]: {n:>4}  '
                  f'{"#"*int(n*40/len(ages))}')
        n_over_500 = sum(1 for a in ages if a > 500)
        if n_over_500:
            print(f'    age >500:     {n_over_500}')

    # Also: per-step phase classification, useful to see if drain steps use older anchors
    print(f'\n=== Anchor age at key steps (sample every 10) ===')
    for entry in age_by_step[::10][:30]:
        step, age, n_rs = entry
        print(f'  step {step:>3} (|RS|={n_rs}): age={age}  '
              f'(fraction: {age/n_rs*100:.0f}% of history)')
    if age_by_step:
        # Last 10 steps before drain
        print(f'\n  Last 10 phase-1b actions in path:')
        for entry in age_by_step[-10:]:
            step, age, n_rs = entry
            print(f'    step {step:>3} (|RS|={n_rs}): age={age}')


if __name__ == '__main__':
    sys.exit(main())
