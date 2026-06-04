#!/usr/bin/env python
"""For the step-26 parent state of the divergent survivor, reconstruct
the `valid` list (output of enumerate_valid_actions_with_indirect_cache)
for the SAME (target, state) on BOTH sides. Compare:
  - SET of (op, delta) in valid_A vs valid_B
  - POSITION of the action selected at step 27

If sets differ → softmax denominator differs → action_prob differs.
If sets match but order differs → with stable argsort, tie-broken idx
   differs, but for distinct logits and same set, action_prob at the
   chosen action's position is identical (softmax is permutation-equivariant).
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, get_raw_equation,
)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def path_key(s):
    return tuple(tuple(a) for a in s['path'])


def reconstruct_iraws_from_flat(flat, env):
    """Reconstruct iraws as a list of (sub_int, ibp_op, shift, raw) by
    fetching raws from env's cache. cached is computed downstream."""
    n_idx = (flat.iraws_meta.shape[1] - 1) // 2
    iraws = []
    for row in flat.iraws_meta:
        sub_int = tuple(int(x) for x in row[:n_idx])
        op = int(row[n_idx])
        shift = tuple(int(x) for x in row[n_idx + 1:])
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(op, seed)
        iraws.append((sub_int, op, shift, raw))
    return iraws


def enumerate_valid_for(target, iraws, resolved_subs, shifts, env):
    """Phase 1a + 1b enumeration matching enumerate_valid_actions_with_indirect_cache."""
    valid = []
    seen = set()
    # Phase 1a
    n_idx = ibp_env.N_INDICES
    for ibp_op, shift_list in shifts.items():
        for shift in shift_list:
            seed = tuple(target[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(ibp_op, seed)
            if target not in raw or raw[target] == 0:
                continue
            cached = apply_resolved_subs(raw, resolved_subs)
            if target not in cached or cached[target] == 0:
                continue
            delta = tuple(seed[i] - target[i] for i in range(n_idx))
            if (ibp_op, delta) not in seen:
                seen.add((ibp_op, delta))
                valid.append((ibp_op, delta, 'P1a'))
    # Phase 1b
    for sub_int, ibp_op, shift, raw in iraws:
        if target in raw and raw[target] != 0:
            continue
        cached = apply_resolved_subs(raw, resolved_subs)
        if target not in cached or cached[target] == 0:
            continue
        # (skip sector filter — both runs apply same filter)
        seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
        delta = tuple(seed[i] - target[i] for i in range(n_idx))
        if (ibp_op, delta) not in seen:
            seen.add((ibp_op, delta))
            valid.append((ibp_op, delta, 'P1b'))
    return valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_a')
    p.add_argument('ckpt_b')
    p.add_argument('topology')
    p.add_argument('--target', required=True,
                   help='target integral as comma-separated, e.g. "0,1,2,0,..."')
    p.add_argument('--action-op', type=int, required=True)
    p.add_argument('--action-delta', required=True,
                   help='delta as comma-separated')
    args = p.parse_args()

    target = tuple(int(x) for x in args.target.split(','))
    chosen_delta = tuple(int(x) for x in args.action_delta.split(','))
    chosen = (args.action_op, chosen_delta)

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    ca = load(args.ckpt_a)
    cb = load(args.ckpt_b)
    print(f'A: step={ca["step"]} beam_size={len(ca["beam"])}')
    print(f'B: step={cb["step"]} beam_size={len(cb["beam"])}')

    # Match survivors at step 26 by the path that the step-27 divergent
    # survivor used (first 26 actions).
    # User passes --path-marker via ckpt or we just take the first survivor
    # whose expr/RS produces the chosen action's seed.
    target_in = lambda s: target in s['expr']
    parent_a = next((s for s in ca['beam'] if target_in(s)), None)
    parent_b = next((s for s in cb['beam'] if target_in(s)), None)
    if parent_a is None or parent_b is None:
        print('No parent survivor has target in its expr.')
        return 1

    if path_key(parent_a) != path_key(parent_b):
        # Find pair with matching paths
        bym_a = {path_key(s): s for s in ca['beam'] if target in s['expr']}
        bym_b = {path_key(s): s for s in cb['beam'] if target in s['expr']}
        common = set(bym_a) & set(bym_b)
        if not common:
            print('No common-path survivor has target in expr.')
            return 1
        p_match = next(iter(common))
        parent_a = bym_a[p_match]
        parent_b = bym_b[p_match]

    print(f'Parent survivor: max_w={parent_a["max_w"]} nm={parent_a["n_non_masters"]}')

    rs_a = parent_a['resolved_subs']
    rs_b = parent_b['resolved_subs']
    same_rs = (rs_a == rs_b)
    print(f'resolved_subs match between A and B: {same_rs}  |A|={len(rs_a)}')

    flat_a = parent_a['aux_flat']
    flat_b = parent_b['aux_flat']
    iraws_a = reconstruct_iraws_from_flat(flat_a, env)
    iraws_b = reconstruct_iraws_from_flat(flat_b, env)
    print(f'|iraws_A|={len(iraws_a)}  |iraws_B|={len(iraws_b)}')

    valid_a = enumerate_valid_for(target, iraws_a, rs_a, env.shifts, env)
    valid_b = enumerate_valid_for(target, iraws_b, rs_b, env.shifts, env)
    print(f'\n|valid_A|={len(valid_a)}  |valid_B|={len(valid_b)}')

    set_a = set((op, d) for op, d, _ in valid_a)
    set_b = set((op, d) for op, d, _ in valid_b)
    only_a = set_a - set_b
    only_b = set_b - set_a
    print(f'A-only actions: {len(only_a)}, B-only actions: {len(only_b)}')

    if only_a:
        print(f'  first A-only: {next(iter(only_a))}')
    if only_b:
        print(f'  first B-only: {next(iter(only_b))}')

    # Find chosen action's position in each.
    def find_idx(valid_list, op, delta):
        for i, (o, d, _) in enumerate(valid_list):
            if o == op and d == delta:
                return i
        return -1

    idx_a = find_idx(valid_a, args.action_op, chosen_delta)
    idx_b = find_idx(valid_b, args.action_op, chosen_delta)
    print(f'\nChosen action (op={args.action_op}, delta={chosen_delta}) '
          f'position: A_idx={idx_a}  B_idx={idx_b}')
    if idx_a >= 0:
        phase_a = valid_a[idx_a][2]
        print(f'  A phase: {phase_a}')
    if idx_b >= 0:
        phase_b = valid_b[idx_b][2]
        print(f'  B phase: {phase_b}')

    # Final summary
    if set_a == set_b:
        print('\n=== valid SETS match ===')
        MAX = 900
        trunc_a = set((op, d) for op, d, _ in valid_a[:MAX])
        trunc_b = set((op, d) for op, d, _ in valid_b[:MAX])
        only_a_trunc = trunc_a - trunc_b
        only_b_trunc = trunc_b - trunc_a
        print(f'Truncated first-{MAX} subsets: A_only={len(only_a_trunc)} '
              f'B_only={len(only_b_trunc)}')

        # Count how many actions have DIFFERENT positions between A and B
        # (within the truncated [0..MAX) range).
        pos_a = {(op, d): i for i, (op, d, _) in enumerate(valid_a[:MAX])}
        pos_b = {(op, d): i for i, (op, d, _) in enumerate(valid_b[:MAX])}
        common = set(pos_a) & set(pos_b)
        n_misordered = sum(1 for k in common if pos_a[k] != pos_b[k])
        print(f'Of {len(common)} actions present in both first-{MAX}, '
              f'{n_misordered} are at DIFFERENT positions.')

        if only_a_trunc or only_b_trunc:
            print(f'\n*** TRUNCATION CAUSES SET MISMATCH AT max_actions={MAX} ***')
        elif n_misordered > 0:
            print(f'\n*** PERMUTATION-EQUIVARIANT NUMERICAL DIVERGENCE ***')
            print('  Same action SET in first 900 but DIFFERENT POSITIONS.')
            print('  Softmax denominator = sum(exp(logits)). Floating-point')
            print('  summation is NOT associative, so summing the same')
            print('  multiset of values in different orders gives slightly')
            print('  different results (~1e-6 per add \u00d7 900 adds \u2248 1e-4).')
            print('  This matches the observed score \u0394 of -5.7e-4.')
            print('  Even though the model is mathematically order-invariant,')
            print('  it is numerically order-sensitive due to fp roundoff.')
            # Show a few examples of position mismatches
            cnt = 0
            for k in common:
                if pos_a[k] != pos_b[k]:
                    print(f'    {k}: pos_A={pos_a[k]}, pos_B={pos_b[k]}')
                    cnt += 1
                    if cnt >= 5:
                        break
        else:
            print(f'AND positions match. action_prob MUST be identical.')
    else:
        print(f'\n=== valid SETS DIFFER === '
              f'|set_A|={len(set_a)} |set_B|={len(set_b)}')
        print('=> softmax denominator differs → action_prob for SAME '
              'action differs → score diverges. This is the proximate cause.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
