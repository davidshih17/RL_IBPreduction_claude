#!/usr/bin/env python
"""Local replay-twice diff: walk the SAME survivor path with BOTH
v4-rescue iraws code and baseline iraws code, comparing iraws at every
step. First step where they differ in action SET (not just order) is the
bug location.

Runs on the login node — no Condor, no model. Replay walks the path with
_apply_action; we materialize iraws via both code paths at each step and
diff. Should complete in 1-2 minutes.

Usage:
  local_replay_diff.py <thin_ckpt> <rank> <topology_dir>
"""
import argparse
import gzip
import os
import pickle
import sys
import time

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology,
    apply_resolved_subs, solve_ibp_for, apply_substitution_target_only,
    add_sub_to_resolved, integral_in_exact_sector,
    compute_indirect_substituted_with_aux,
    compute_indirect_substituted_exprkeyed_delta,
    IBPEnvironment,
)


def load_ckpt(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt', help='thin checkpoint .ckpt.r1[.suffix]')
    p.add_argument('topology', help='topology dir')
    p.add_argument('--rank', type=int, default=0,
                   help='which beam survivor to walk (default 0)')
    p.add_argument('--max-step', type=int, default=None,
                   help='stop at this step (default: full path)')
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--start-step', type=int, default=0,
                   help='start verbose comparison from this step (default 0)')
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    ibp_t = env.ibp_t
    li_t = env.li_t
    shifts = env.shifts

    raw_eq_cache_v4 = {}
    raw_eq_cache_base = {}

    d = load_ckpt(args.ckpt)
    target_sector = tuple(d['target_sector'])
    start_expr = dict(d['start_expr'])
    beam = sorted(d['beam'],
                  key=lambda s: (tuple(s['max_w']), int(s['n_non_masters']),
                                 -float(s['score'])))
    if args.rank >= len(beam):
        print(f'rank {args.rank} out of range (beam has {len(beam)})')
        return 1
    survivor = beam[args.rank]
    path = list(survivor['path'])
    max_step = args.max_step or len(path)
    print(f'Replaying rank {args.rank} survivor: max_w={survivor["max_w"]} '
          f'nm={survivor["n_non_masters"]} path_len={len(path)} '
          f'replaying through step {max_step}', flush=True)
    print(f'target_sector={target_sector}', flush=True)
    print(f'start_expr={start_expr}', flush=True)

    # Walk path step-by-step, computing v4 and baseline iraws at each step.
    # State after action k = path[0..k] applied.
    expr_v4 = dict(start_expr)
    rs_v4 = {}
    prev_aux_v4 = ([], [], {}, [])  # (cu, ubm, rid, iraws) for v4

    t0 = time.time()
    for step, (target, ibp_op, delta_shift) in enumerate(path[:max_step], 1):
        # Apply action to advance expr + RS (same for both modes).
        seed = tuple(target[i] + delta_shift[i] for i in range(ibp_env.N_INDICES))
        # Get raw via the raw_eq_cache (shared cache key (op, seed)).
        if (ibp_op, seed) not in raw_eq_cache_v4:
            raw = ibp_env.get_raw_equation(ibp_t, li_t,
                                            ibp_op, seed)
            raw_eq_cache_v4[(ibp_op, seed)] = raw
        raw = raw_eq_cache_v4[(ibp_op, seed)]
        cached = apply_resolved_subs(raw, rs_v4)
        if target not in cached or cached[target] == 0:
            print(f'step {step}: action invalid (target NOT IN CACHED). Stop.')
            print(f'  target={target}  ibp_op={ibp_op} seed={seed}')
            print(f'  raw size={len(raw)} cached size={len(cached)} rs_v4 size={len(rs_v4)}')
            break
        sol = solve_ibp_for(cached, target)
        if sol is None:
            print(f'step {step}: sol is None. Stop.')
            break
        sol_target = {k: v for k, v in sol.items()
                      if integral_in_exact_sector(k, target_sector)}
        new_expr = apply_substitution_target_only(expr_v4, target, sol_target, target_sector)
        new_rs = add_sub_to_resolved(rs_v4, target, sol_target)
        new_resolved_sol = apply_resolved_subs(sol_target, rs_v4)

        # Build v4-rescue iraws (exprkeyed_delta).
        from beam_search_utils import get_non_masters
        expr_nm = get_non_masters(new_expr, target_sector)
        result_v4, aux_v4 = compute_indirect_substituted_exprkeyed_delta(
            prev_aux_v4, expr_nm.keys(),
            target, new_resolved_sol, new_rs,
            ibp_t, li_t, shifts, raw_eq_cache_v4,
            target_sector=None,
        )

        # Build baseline iraws (from-scratch).
        subs = list(new_rs.keys())
        result_base, _ = compute_indirect_substituted_with_aux(
            subs, new_rs, ibp_t, li_t, shifts,
            raw_eq_cache_base, target_sector=None,
        )

        # Index both by (op, seed) for comparison.
        def index(result):
            idx = {}
            for entry in result:
                sub_int, op, sh, raw_e, cached_e, ubm = entry
                seed_e = tuple(sub_int[i] - sh[i] for i in range(ibp_env.N_INDICES))
                if (op, seed_e) not in idx:
                    idx[(op, seed_e)] = (cached_e, raw_e)
            return idx

        idx_v4 = index(result_v4)
        idx_base = index(result_base)

        keys_v4 = set(idx_v4)
        keys_base = set(idx_base)
        missed_v4 = keys_base - keys_v4
        extra_v4 = keys_v4 - keys_base
        common = keys_v4 & keys_base

        cached_diff = []
        for k in common:
            if idx_v4[k][0] != idx_base[k][0]:
                cached_diff.append(k)

        # Compute action set for each target in expr_nm (Phase 1b).
        targets = list(expr_nm.keys())

        def action_set(idx):
            actions = set()
            for (op, seed_e), (cached_e, raw_e) in idx.items():
                for T in targets:
                    if T in raw_e and raw_e[T] != 0:
                        continue
                    if T not in cached_e or cached_e[T] == 0:
                        continue
                    delta = tuple(seed_e[i] - T[i] for i in range(ibp_env.N_INDICES))
                    actions.add((op, delta, T))
            return actions

        actions_v4 = action_set(idx_v4)
        actions_base = action_set(idx_base)
        action_missed = actions_base - actions_v4
        action_extra = actions_v4 - actions_base

        if step >= args.start_step:
            print(f'step={step:3d} expr|={len(new_expr)} expr_nm|={len(targets)} '
                  f'iraws_v4={len(result_v4)} iraws_base={len(result_base)} '
                  f'cached_diff={len(cached_diff)} '
                  f'action_missed={len(action_missed)} action_extra={len(action_extra)}',
                  flush=True)

        if action_missed or action_extra:
            print(f'*** DIVERGENCE at step {step}: missed={len(action_missed)} '
                  f'extra={len(action_extra)} ***', flush=True)
            for i, a in enumerate(list(action_missed)[:3]):
                print(f'  baseline-only #{i}: op={a[0]} delta={a[1]} target={a[2]}')
            for i, a in enumerate(list(action_extra)[:3]):
                print(f'  v4-only #{i}: op={a[0]} delta={a[1]} target={a[2]}')
            print(f'  v4 expr at step {step}: {len(new_expr)} terms')
            print(f'  v4 expr_nm: {len(targets)} terms')
            print(f'  RS size: {len(new_rs)}')
            return 1

        expr_v4 = new_expr
        rs_v4 = new_rs
        prev_aux_v4 = aux_v4

    elapsed = time.time() - t0
    print(f'\nNO action divergence through step {max_step} ({elapsed:.1f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
