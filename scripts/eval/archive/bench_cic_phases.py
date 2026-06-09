"""
Decompose compute_indirect_substituted into its 5 phases at production-like
state sizes. Uses the BEAM_PROFILE_CIC=1 mechanism already in the function:
each call appends to ibp_env._CIC_PROFILE.

Phases (from the function body):
  1. collect       walk subs, expand into (sub_int, ibp_op, shift, raw) tuples
  2. dedup         build raw_id_to_idx (unique raws)
  3. apply_subs    apply_resolved_subs_batch(unique_raws, resolved_subs)  ← heavy
  4. union_bm      compute OR-union sector bitmasks per cached raw
  5. build         build the result list
"""
import os
import random
import sys
from pathlib import Path

os.environ['BEAM_PROFILE_CIC'] = '1'  # MUST be set before import to take effect

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    add_sub_to_resolved,
    compute_indirect_substituted,
)


def random_tarsect_int(rng):
    out = []
    for i in range(ibp_env.N_INDICES):
        out.append(rng.randint(1, 3) if i < 8 else rng.randint(-3, 0))
    return tuple(out)


def random_sol(rng, n_terms):
    return {random_tarsect_int(rng): rng.randint(1, ibp_env.PRIME - 1)
            for _ in range(n_terms)}


def grow_state(env, N, seed=42):
    rng = random.Random(seed)
    subs = {}
    resolved_subs = {}
    for _ in range(N):
        target = random_tarsect_int(rng)
        sol = random_sol(rng, rng.randint(8, 22))
        subs[target] = sol
        resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)
    return subs, resolved_subs


def measure(env, N_subs, n_warm=2, n_runs=5):
    subs, resolved_subs = grow_state(env, N_subs)

    # Warm
    for _ in range(n_warm):
        compute_indirect_substituted(
            subs, resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )
    # Clear the profile list and run n_runs
    ibp_env._CIC_PROFILE = []
    for _ in range(n_runs):
        compute_indirect_substituted(
            subs, resolved_subs, env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        )

    # Average the per-run profiles
    rows = ibp_env._CIC_PROFILE
    n = len(rows)
    avg = {}
    keys = ['t_total', 't_phase1_collect', 't_phase2_dedup', 't_phase3_apply_subs',
            't_phase4_union_bm', 't_phase5_build', 'n_indirect_raws', 'n_unique_raws',
            'n_sub_cache_misses']
    for k in keys:
        avg[k] = sum(r.get(k, 0) for r in rows) / n
    return avg


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    print('=' * 115)
    print('compute_indirect_substituted: per-phase ms breakdown averaged over 5 calls')
    print('=' * 115)
    print(f'{"|subs|":>7} {"raws":>7} {"unique":>7} '
          f'{"total":>8} {"P1 collect":>11} {"P2 dedup":>9} {"P3 apply":>10} '
          f'{"P4 union_bm":>12} {"P5 build":>9} {"P3 %":>6}')
    print('-' * 115)
    for N in [50, 100, 150, 200, 250, 300]:
        r = measure(env, N)
        total_ms = 1000 * r['t_total']
        p3_pct = 100 * r['t_phase3_apply_subs'] / r['t_total'] if r['t_total'] else 0
        print(f'{N:>7} {int(r["n_indirect_raws"]):>7} {int(r["n_unique_raws"]):>7} '
              f'{total_ms:>7.1f}ms '
              f'{1000*r["t_phase1_collect"]:>10.1f}ms '
              f'{1000*r["t_phase2_dedup"]:>8.1f}ms '
              f'{1000*r["t_phase3_apply_subs"]:>9.1f}ms '
              f'{1000*r["t_phase4_union_bm"]:>11.1f}ms '
              f'{1000*r["t_phase5_build"]:>8.1f}ms '
              f'{p3_pct:>5.1f}%')
    print('=' * 115)
