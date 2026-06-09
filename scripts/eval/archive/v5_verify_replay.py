#!/usr/bin/env python
"""Verify v5 mathematically: replay the v5 beam's best-state path against the
FULL start_expr (no stripping). The result should:
  1. Be a valid expression (sum of integrals with coefficients)
  2. Have NO active-weight target-sector non-masters left
     (because v5 said it drained those)
  3. Contain only passenger-weight terms + masters + sub-sector content

If (2) fails, v5 has a bug: it claimed to drain active content but the real
full reduction still has active stuff.
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
    apply_all_substitutions, apply_substitution, solve_ibp_for,
    weight, is_master,
)
from beam_search_utils import get_non_masters, get_sector_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True,
                   help='Pickle from beam_search_v5 --output (has best_state.path)')
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    start_int = tuple(int(x) for x in args.integral.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))
    print(f'start_int={start_int}  weight={start_w}  '
          f'active threshold (w1,w2) >= {start_w12}')

    with open(args.ckpt, 'rb') as f:
        d = pickle.load(f)
    path = d['best_state']['path']
    print(f'Loaded v5 best_state path of length {len(path)}')

    # Replay path against the full start_expr (no stripping). Continue past
    # failures by reporting them and not advancing subs/expr for that step.
    expr = {start_int: 1}
    subs = {}
    print(f'\nReplaying {len(path)} actions against full start_expr...')
    print(f'{"step":>4} {"target_w":>10} {"status":>8} {"|expr|":>7} '
          f'{"|active_nm|":>11} {"|pass_nm|":>10} {"max_w_active_nm":>18}')
    n_failed = 0
    for step, (target, ibp_op, delta) in enumerate(path):
        seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        cached = apply_all_substitutions(raw, subs)
        target_w = (weight(target)[0], weight(target)[1])
        status = 'OK'
        if target not in cached or cached[target] == 0:
            status = 'NO_TGT'
            n_failed += 1
        else:
            sol = solve_ibp_for(cached, target)
            if sol is None:
                status = 'NO_SOL'
                n_failed += 1
            else:
                subs[target] = sol
                expr = apply_substitution(expr, target, sol)

        # Report current expr state (target-sector, active non-masters only)
        n_active_nm = 0
        n_pass_nm = 0
        max_w_active = (0, 0)
        for k in expr:
            sec = tuple(get_sector_mask(k))
            if sec != target_sector:
                continue
            if is_master(k):
                continue
            w12 = (weight(k)[0], weight(k)[1])
            if w12 >= start_w12:
                n_active_nm += 1
                if w12 > max_w_active:
                    max_w_active = w12
            else:
                n_pass_nm += 1
        print(f'{step+1:>4} {str(target_w):>10} {status:>8} {len(expr):>7} '
              f'{n_active_nm:>11} {n_pass_nm:>10} {str(max_w_active):>18}')

    print(f'\nFinal full expr after {len(path)}-step replay:')
    print(f'  total entries:      {len(expr)}')

    # Classify each entry
    n_target_sector = 0
    n_subsector = 0
    n_active_target_nm = 0       # active, target sector, non-master
    n_active_target_master = 0   # active, target sector, master
    n_passenger_target_nm = 0    # passenger weight, target sector, non-master
    n_passenger_target_master = 0
    nd = ibp_env.N_DENOMINATORS
    samples_active_target_nm = []

    for k in expr:
        sec = tuple(get_sector_mask(k))
        in_target_sector = (sec == target_sector)
        if in_target_sector:
            n_target_sector += 1
        else:
            n_subsector += 1
            continue
        w = weight(k)
        active = (w[0], w[1]) >= start_w12
        master = is_master(k)
        if active and not master:
            n_active_target_nm += 1
            if len(samples_active_target_nm) < 10:
                samples_active_target_nm.append((k, w[:2]))
        elif active and master:
            n_active_target_master += 1
        elif not active and not master:
            n_passenger_target_nm += 1
        elif not active and master:
            n_passenger_target_master += 1

    print(f'  target-sector:      {n_target_sector}')
    print(f'    active non-master:    {n_active_target_nm}  <-- v5 claims this is 0')
    print(f'    active master:        {n_active_target_master}')
    print(f'    passenger non-master: {n_passenger_target_nm}')
    print(f'    passenger master:     {n_passenger_target_master}')
    print(f'  sub-sector:         {n_subsector}')

    # Weight histogram of all target-sector entries
    w_counts = Counter()
    for k in expr:
        if tuple(get_sector_mask(k)) == target_sector:
            w_counts[(weight(k)[0], weight(k)[1])] += 1
    print(f'\n  weight histogram (target-sector):')
    for w, c in sorted(w_counts.items(), reverse=True):
        marker = ' <-- ACTIVE (≥ start)' if w >= start_w12 else '  (passenger)'
        print(f'    (w1,w2)={w}  count={c}{marker}')

    print(f'\n=== VERDICT ===')
    if n_active_target_nm > 0:
        print(f'BUG: v5 claimed active drained but replay still has '
              f'{n_active_target_nm} active non-masters in target sector')
        print(f'Sample active non-masters left:')
        for k, w12 in samples_active_target_nm:
            print(f'  {k}  weight(w1,w2)={w12}')
        return 1
    else:
        print(f'OK: replay confirms 0 active target-sector non-masters remain.')
        print(f'    start_int has been reduced to: '
              f'masters + passenger non-masters + sub-sector content.')
        print(f'    The {n_passenger_target_nm} passenger non-masters still '
              f'need their own reductions to reach a pure master expansion.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
