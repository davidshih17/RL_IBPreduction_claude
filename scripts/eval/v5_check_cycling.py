#!/usr/bin/env python
"""Replay each beam[i].path step-by-step in v5 semantics. At each step
record a fingerprint of expr (and of resolved_subs keys). Report any
fingerprint repeats — that's evidence of cycling within a single path.

Also check ACROSS beam states: how many distinct (expr) fingerprints exist
in the final beam.
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
    apply_resolved_subs, solve_ibp_for, weight, is_master,
)
from beam_search_v5 import (
    apply_substitution_v5, add_sub_to_resolved_v5, State_v5,
)
from beam_search_utils import get_sector_mask, get_non_masters


def expr_fp(expr):
    return frozenset(expr.items())


def rs_keys_fp(rs):
    return frozenset(rs.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--n-paths', type=int, default=5,
                   help='how many beam[i] paths to trace')
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

    with open(args.ckpt, 'rb') as f:
        d = pickle.load(f)

    # Check across-beam diversity at the final state
    final_beam = d['beam']
    expr_fps_final = Counter()
    for s in final_beam:
        expr_fps_final[expr_fp(s['expr'])] += 1
    print(f'Final beam size: {len(final_beam)}')
    print(f'Distinct expr fingerprints in final beam: {len(expr_fps_final)}')
    if len(expr_fps_final) < len(final_beam):
        print(f'  Duplicates exist! Most common expr appears '
              f'{max(expr_fps_final.values())} times.')

    # Replay each beam[i].path and track per-step expr fingerprint
    print(f'\nReplaying {args.n_paths} beam[i] paths to check within-path cycles:')
    for bi in range(min(args.n_paths, len(final_beam))):
        path = final_beam[bi]['path']
        # Reconstruct path step by step in v5 semantics
        state = State_v5(
            expr={start_int: 1},
            resolved_subs={},
            sub_accum={},
            score=0.0,
            path=[],
            n_non_masters=1,
        )
        seen_expr = {}  # fingerprint -> step at which first seen
        seen_rs_keys = {}
        revisits = []
        for step, (target, op, delta) in enumerate(path):
            seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
            raw = env.get_raw_equation_cached(op, seed)
            cached = apply_resolved_subs(raw, state.resolved_subs)
            if target not in cached or cached[target] == 0:
                print(f'  beam[{bi}] step {step+1}: replay failed')
                break
            sol = solve_ibp_for(cached, target)
            new_expr, new_sub_accum = apply_substitution_v5(
                state.expr, state.sub_accum, target, sol, target_sector,
                start_w12,
            )
            new_rs = add_sub_to_resolved_v5(
                state.resolved_subs, target, sol, start_w12,
            )
            nm = get_non_masters(new_expr, target_sector)
            state = State_v5(
                expr=new_expr, resolved_subs=new_rs,
                sub_accum=new_sub_accum,
                score=0.0, path=[], n_non_masters=len(nm),
            )
            fp = expr_fp(new_expr)
            rfp = rs_keys_fp(new_rs)
            if fp in seen_expr:
                revisits.append((step+1, seen_expr[fp], fp))
            else:
                seen_expr[fp] = step + 1
            if rfp in seen_rs_keys and seen_rs_keys[rfp] != step + 1:
                pass  # RS keys only grow, so this won't trigger
            seen_rs_keys[rfp] = step + 1
        print(f'  beam[{bi}]: path_len={len(path)}  '
              f'distinct expr fps={len(seen_expr)}  '
              f'expr revisits={len(revisits)}')
        if revisits[:3]:
            for cur_step, first_step, fp in revisits[:3]:
                # Describe the expr briefly
                items = list(fp)
                print(f'    revisit at step {cur_step}: same expr as step {first_step}  '
                      f'({len(items)} terms)')

if __name__ == '__main__':
    sys.exit(main())
