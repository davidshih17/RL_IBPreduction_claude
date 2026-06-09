#!/usr/bin/env python3
"""Replay dedup-OFF baseline path step-by-step. Hash target-sector expr at
each step. Detect revisits along the winning trajectory.
"""
import argparse
import hashlib
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env  # access module attrs dynamically (set by init_from_topology)
from sailir.ibp_env import IBPEnvironment, set_prime
from beam_search_utils import get_sector_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    parser.add_argument('--topology', required=True)
    parser.add_argument('--prime', type=int, default=1009)
    args = parser.parse_args()

    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    env = IBPEnvironment()

    # Get fresh references AFTER init_from_topology has populated module state.
    from sailir.ibp_env import (
        get_raw_equation, apply_resolved_subs, solve_ibp_for,
        add_sub_to_resolved, apply_substitution,
        integral_in_exact_sector, is_master, weight,
    )
    N_INDICES = ibp_env.N_INDICES
    print(f"N_INDICES from module: {N_INDICES}")

    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    integral = result['original_integral']
    path = result['path']
    target_sector = tuple(get_sector_mask(integral))
    print(f"Replaying {len(path)} steps for I{list(integral)} target_sector={target_sector}")
    print()

    full_expr = {integral: 1}
    full_rs = {}

    def expr_t_hash(full_expr):
        targ = {k: v for k, v in full_expr.items()
                if integral_in_exact_sector(k, target_sector)}
        canon = repr(sorted(targ.items()))
        return hashlib.md5(canon.encode()).hexdigest()[:10], targ

    def stats(targ):
        nms = [k for k in targ if not is_master(k)]
        nm = len(nms)
        if nm == 0:
            mw = (0, 0)
        else:
            mw = max((weight(k)[0], weight(k)[1]) for k in nms)
        return nm, mw

    visited = {}
    revisits = []
    h, t = expr_t_hash(full_expr)
    nm, mw = stats(t)
    visited[h] = -1
    print(f"step=-1 (initial): hash={h} nm={nm} mw={mw}")

    for step, (target, ibp_op, delta) in enumerate(path):
        try:
            seed = tuple(target[i] + delta[i] for i in range(N_INDICES))
        except Exception as e:
            print(f"step {step}: seed construction failed: {e}; target={target} delta={delta}")
            break
        raw = get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
        cached = apply_resolved_subs(raw, full_rs)
        if target not in cached or cached[target] == 0:
            continue
        sol = solve_ibp_for(cached, target)
        if sol is None:
            continue
        full_rs = add_sub_to_resolved(full_rs, target, sol)
        full_expr = apply_substitution(full_expr, target, sol)
        h, t = expr_t_hash(full_expr)
        nm, mw = stats(t)
        if h in visited:
            revisits.append((step, visited[h], h, nm, mw))
        else:
            visited[h] = step

    print()
    print(f"Replayed {len(path)} steps")
    print(f"Distinct target-sector expr hashes on winning path: {len(visited)}")
    print(f"Revisits on winning path: {len(revisits)}")
    if revisits:
        print("\nFirst 20 revisits (step, prev_step, hash, nm, mw):")
        for r in revisits[:20]:
            print(f"  step={r[0]:4d} revisits hash={r[2]} from step={r[1]:4d} (Δ={r[0]-r[1]:3d}) nm={r[3]:3d} mw={r[4]}")
        print(f"\nLast 20 revisits:")
        for r in revisits[-20:]:
            print(f"  step={r[0]:4d} revisits hash={r[2]} from step={r[1]:4d} (Δ={r[0]-r[1]:3d}) nm={r[3]:3d} mw={r[4]}")
    else:
        print("\nNO revisits — winning path has 360 distinct target-sector exprs.")


if __name__ == '__main__':
    main()
