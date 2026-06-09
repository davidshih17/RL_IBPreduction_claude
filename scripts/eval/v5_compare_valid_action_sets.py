#!/usr/bin/env python
"""Head-to-head comparison of valid-action enumeration between
baseline (full RS values, including passenger content) and v5-style
(RS values stripped of integrals with weight below start_w12).

Procedure:
  1. Load a baseline thick checkpoint (has full expr, subs, resolved_subs).
  2. For each survivor, for each tied target:
     a. enumerate_valid_actions_with_indirect_cache with FULL RS
     b. Strip RS values of passenger weight (and the corresponding subs dummy
        keys), and re-enumerate
  3. Compare the (op, delta) sets.

If user's theory is right: the two sets are IDENTICAL for every (state, target).
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
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, compute_indirect_substituted, weight, is_master,
)
from beam_search_utils import get_non_masters, get_sector_mask


def is_active(integral, start_w12):
    w = weight(integral)
    return (w[0], w[1]) >= start_w12


def strip_passenger(d, start_w12):
    return {k: v for k, v in d.items() if is_active(k, start_w12)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Baseline thick ckpt (.pkl)')
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--max-survivors', type=int, default=10)
    p.add_argument('--max-targets-per-survivor', type=int, default=2)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    start_int = tuple(int(x.strip("'\"")) for x in args.integral.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))
    print(f'Start integral weight = {start_w}, active = (w1,w2) >= {start_w12}')

    with open(args.ckpt, 'rb') as f:
        c = pickle.load(f)
    print(f'Loaded ckpt at step {c.get("step", "?")} with {len(c["beam"])} survivors')

    n_idx = ibp_env.N_INDICES
    total_state_target_pairs = 0
    total_baseline_only = 0
    total_v5_only = 0
    total_both = 0
    perfect_matches = 0
    mismatches = []

    for si, s in enumerate(c['beam'][:args.max_survivors]):
        expr = s['expr']
        subs_full = s['subs']
        rs_full = s['resolved_subs']

        # Compute v5-stripped versions
        # subs keys are same as RS keys (past targets in baseline). All past targets
        # at (8,4) phase are at active weight already (per our discussion). But
        # the VALUES of subs[K] and RS[K] in baseline may contain passenger content.
        # v5: strip passenger from each value dict.
        subs_v5 = {k: strip_passenger(v, start_w12) for k, v in subs_full.items()}
        rs_v5 = {k: strip_passenger(v, start_w12) for k, v in rs_full.items()}

        # Also verify subs/RS keys themselves are all at active weight (sanity)
        non_active_keys = [k for k in subs_full.keys() if not is_active(k, start_w12)]
        if non_active_keys:
            print(f'  survivor {si}: WARNING — {len(non_active_keys)} subs keys at '
                  f'passenger weight (e.g. {non_active_keys[0]} weight='
                  f'{(weight(non_active_keys[0])[0], weight(non_active_keys[0])[1])})')

        nm = get_non_masters(expr, target_sector)
        if not nm:
            continue
        mw = max((weight(k)[0], weight(k)[1]) for k in nm)
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        tied = tied[:args.max_targets_per_survivor]

        for target in tied:
            total_state_target_pairs += 1

            # BASELINE enumeration with full subs/RS
            ic_full = compute_indirect_substituted(
                subs_full, rs_full, env.ibp_t, env.li_t, env.shifts,
                env._raw_eq_cache,
            )
            va_full = enumerate_valid_actions_with_indirect_cache(
                target, ic_full, subs_full, rs_full,
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )

            # V5 enumeration with stripped subs/RS
            ic_v5 = compute_indirect_substituted(
                subs_v5, rs_v5, env.ibp_t, env.li_t, env.shifts,
                env._raw_eq_cache,
            )
            va_v5 = enumerate_valid_actions_with_indirect_cache(
                target, ic_v5, subs_v5, rs_v5,
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )

            set_full = set(va_full)
            set_v5 = set(va_v5)
            both = set_full & set_v5
            only_full = set_full - set_v5
            only_v5 = set_v5 - set_full

            total_both += len(both)
            total_baseline_only += len(only_full)
            total_v5_only += len(only_v5)

            if not only_full and not only_v5:
                perfect_matches += 1
            else:
                mismatches.append({
                    'si': si, 'target': target,
                    'len_full': len(va_full), 'len_v5': len(va_v5),
                    'only_full': only_full, 'only_v5': only_v5,
                })

            if total_state_target_pairs <= 5:
                print(f'  survivor {si} target {target}: '
                      f'|baseline|={len(va_full)} |v5|={len(va_v5)} '
                      f'shared={len(both)} only_baseline={len(only_full)} '
                      f'only_v5={len(only_v5)} '
                      f'{"MATCH" if (not only_full and not only_v5) else "DIFFER"}')

    print(f'\n=== SUMMARY ===')
    print(f'Total (state, target) pairs tested: {total_state_target_pairs}')
    print(f'Perfect matches (sets equal):       {perfect_matches}/{total_state_target_pairs}')
    print(f'Total shared actions:               {total_both}')
    print(f'Total baseline-only:                {total_baseline_only}')
    print(f'Total v5-only:                      {total_v5_only}')
    if mismatches:
        print(f'\nFirst 3 mismatches:')
        for m in mismatches[:3]:
            print(f'  survivor {m["si"]} target {m["target"]}: '
                  f'baseline={m["len_full"]} v5={m["len_v5"]} '
                  f'only_baseline={len(m["only_full"])} only_v5={len(m["only_v5"])}')
            if m['only_full']:
                print(f'    sample only_baseline: {list(m["only_full"])[:2]}')
            if m['only_v5']:
                print(f'    sample only_v5: {list(m["only_v5"])[:2]}')

    # Verdict
    if total_baseline_only == 0 and total_v5_only == 0:
        print(f'\n*** USER\'S THEORY CONFIRMED ***')
        print(f'For every (state, target), the v5-stripped enumeration produces')
        print(f'the IDENTICAL set of (op, delta) actions as baseline\'s full enumeration.')
        print(f'Stripping passenger from RS values does not affect the valid action set.')
    else:
        print(f'\n*** USER\'S THEORY FALSIFIED — action sets diverge ***')


if __name__ == '__main__':
    sys.exit(main())
