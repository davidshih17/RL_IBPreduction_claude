#!/usr/bin/env python
"""Given two thick checkpoints AND a specific survivor with a score
divergence, locate the proximate cause.

Strategy:
  1. Take the divergent survivor's path at step K (where score diverges).
  2. Look at step K-1 ckpts on BOTH sides. Find the same survivor
     (path[:K-1] match). Are their scores identical at K-1?
       - YES: divergence came from action #K-1 (the one taken to reach K).
              Compare the per-action data for that action.
       - NO: recurse — divergence already in ancestor.
  3. For the action that caused the score to diverge:
       - Show its (target, ibp_op, delta) tuple.
       - Look at the parent state's iraws (FlatAux) on both sides:
         specifically the (op, delta) action's POSITION in
         enumerate_valid_actions's `valid` list. If different position
         → torch.argsort tie-break picks different idx → action_prob diff.

Usage:
  diagnose_score_div.py <dir_a> <dir_b> --step K --path-marker "(...)"
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def path_key(s):
    return tuple(tuple(a) for a in s['path'])


def find_survivor_with_score_div(ca, cb):
    """Find first survivor whose path matches but score differs."""
    by_a = {path_key(s): s for s in ca['beam']}
    by_b = {path_key(s): s for s in cb['beam']}
    for p in by_a:
        if p in by_b:
            sa = by_a[p]
            sb = by_b[p]
            if sa['score'] != sb['score']:
                return p, sa, sb
    return None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir_a')
    p.add_argument('dir_b')
    p.add_argument('--max-step', type=int, default=80,
                   help='upper bound on step to search')
    args = p.parse_args()

    # 1. Find first step where any survivor's score diverges with matching path.
    base_a = Path(args.dir_a)
    base_b = Path(args.dir_b)

    divergent = None
    for step in range(1, args.max_step + 1):
        fa = base_a / f'result.pkl.ckpt.r1.step{step:04d}'
        fb = base_b / f'result.pkl.ckpt.r1.step{step:04d}'
        if not (fa.exists() and fb.exists()):
            break
        ca = load(fa)
        cb = load(fb)
        p_div, sa, sb = find_survivor_with_score_div(ca, cb)
        if p_div is not None:
            divergent = (step, p_div, sa, sb)
            print(f'First score-divergent survivor found at step {step}',
                  flush=True)
            break

    if divergent is None:
        print(f'No score divergence in first {args.max_step} steps.')
        return 0

    step, p_div, sa, sb = divergent
    print(f'  score_A = {sa["score"]!r}')
    print(f'  score_B = {sb["score"]!r}')
    print(f'  delta   = {sa["score"] - sb["score"]:+.6e}')
    print(f'  path_len = {len(p_div)}')
    print(f'  last 3 actions: {p_div[-3:]}')

    # 2. Walk back through ancestors. For each prior step K, find the
    #    survivor whose path matches p_div[:K] and compare scores.
    print(f'\nWalking ancestors to locate proximate score divergence...')
    last_match = None  # step where scores still matched
    last_div = step    # step where score diverges
    for K in range(step - 1, 0, -1):
        fa = base_a / f'result.pkl.ckpt.r1.step{K:04d}'
        fb = base_b / f'result.pkl.ckpt.r1.step{K:04d}'
        if not (fa.exists() and fb.exists()):
            continue
        ca = load(fa)
        cb = load(fb)
        prefix = p_div[:K]
        sa_K = {path_key(s): s for s in ca['beam']}.get(prefix)
        sb_K = {path_key(s): s for s in cb['beam']}.get(prefix)
        if sa_K is None or sb_K is None:
            print(f'  step {K}: prefix not in beam (A={sa_K is not None} '
                  f'B={sb_K is not None}); ancestor culled \u2014 stop')
            last_match = None
            break
        if sa_K['score'] == sb_K['score']:
            print(f'  step {K}: scores MATCH ({sa_K["score"]!r})')
            last_match = K
            break
        else:
            print(f'  step {K}: scores DIFFER '
                  f'A={sa_K["score"]!r} B={sb_K["score"]!r} '
                  f'delta={sa_K["score"] - sb_K["score"]:+.6e}')
            last_div = K

    if last_match is None:
        print('\nNo matching ancestor found in scanned range \u2014 divergence '
              'is earlier than any visible step.')
        return 1

    proximate_action_step = last_match + 1  # action taken AFTER last_match
    print(f'\n*** PROXIMATE SCORE DIVERGENCE at action #{proximate_action_step} '
          f'(taken on parent state at step {last_match}, child at step '
          f'{proximate_action_step}) ***')

    # 3. Examine that specific action.
    action = p_div[proximate_action_step - 1]
    target, ibp_op, delta_shift = action
    print(f'  action: target={target}')
    print(f'          ibp_op={ibp_op} delta_shift={delta_shift}')

    # 4. Look at the parent (step last_match) survivor's aux on both sides.
    fa = base_a / f'result.pkl.ckpt.r1.step{last_match:04d}'
    fb = base_b / f'result.pkl.ckpt.r1.step{last_match:04d}'
    ca = load(fa)
    cb = load(fb)
    prefix = p_div[:last_match]
    parent_a = {path_key(s): s for s in ca['beam']}[prefix]
    parent_b = {path_key(s): s for s in cb['beam']}[prefix]

    flat_a = parent_a.get('aux_flat')
    flat_b = parent_b.get('aux_flat')
    if flat_a is None or flat_b is None:
        print(f'\nMissing aux_flat on one side (A={flat_a is not None} '
              f'B={flat_b is not None}); cannot diff iraws.')
        return 1

    print(f'\nParent step-{last_match} aux summary:')
    print(f'  A: n_iraws={flat_a.n_iraws}, n_cu={flat_a.n_cu}')
    print(f'  B: n_iraws={flat_b.n_iraws}, n_cu={flat_b.n_cu}')

    # 5. Convert iraws_meta back to (sub_int, op, shift) tuples and find
    #    which entry corresponds to the action's seed.
    import numpy as np
    n_idx = (flat_a.iraws_meta.shape[1] - 1) // 2

    def entries(flat):
        out = []
        for row in flat.iraws_meta:
            sub_int = tuple(int(x) for x in row[:n_idx])
            op = int(row[n_idx])
            shift = tuple(int(x) for x in row[n_idx + 1:])
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            out.append((sub_int, op, shift, seed))
        return out

    entries_a = entries(flat_a)
    entries_b = entries(flat_b)

    # The action's seed in the parent state = target + delta_shift
    # Then check which iraws entry produces this seed for op=ibp_op.
    action_seed = tuple(target[i] + delta_shift[i] for i in range(n_idx))
    print(f'\nAction seed = target + delta = {action_seed}, ibp_op = {ibp_op}')

    def find_pos(entries, op, seed):
        positions = [(i, sub_int, shift) for i, (sub_int, opx, shift, sd)
                     in enumerate(entries) if opx == op and sd == seed]
        return positions

    pos_a = find_pos(entries_a, ibp_op, action_seed)
    pos_b = find_pos(entries_b, ibp_op, action_seed)
    print(f'  Action found in A at iraws indices: {[p[0] for p in pos_a]}')
    print(f'  Action found in B at iraws indices: {[p[0] for p in pos_b]}')
    if pos_a and pos_b:
        print(f'  A first occurrence at iraws[{pos_a[0][0]}]: '
              f'sub_int={pos_a[0][1]}')
        print(f'  B first occurrence at iraws[{pos_b[0][0]}]: '
              f'sub_int={pos_b[0][1]}')
        if pos_a[0][0] != pos_b[0][0]:
            print('  → iraws iteration POSITION differs for this action. '
                  '\n    enumerate_valid_actions appends in iteration order, '
                  '\n    so the (ibp_op, delta) tuple appears at different '
                  'list index. \n    torch.argsort(stable) tie-break picks '
                  'different `idx` → \n    different probs[i, idx] → different '
                  'action_prob → score diverges.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
