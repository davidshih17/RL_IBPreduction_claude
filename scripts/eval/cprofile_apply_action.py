"""
cProfile breakdown of apply_action_resolved_target_only at a representative
late-state (|subs|=200). Resolves the 73% "residual" lump from the
coarse phase-level profile by attributing every function call.
"""
import cProfile
import io
import pstats
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    get_raw_equation, add_sub_to_resolved,
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
    expr_t = random_sol(rng, 25)
    subs = {}
    resolved_subs = {}
    for _ in range(N):
        target = random_tarsect_int(rng)
        sol = random_sol(rng, rng.randint(8, 22))
        subs[target] = sol
        resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)
    return expr_t, subs, resolved_subs


def find_working_action(env, target, expr_t, subs, resolved_subs):
    """Find an (ibp_op, delta) such that solve_ibp_for would succeed."""
    for op in env.shifts:
        for sh in env.shifts[op]:
            seed = tuple(target[i] - sh[i] for i in range(ibp_env.N_INDICES))
            raw = get_raw_equation(env.ibp_t, env.li_t, op, seed)
            if target in raw and raw[target] != 0:
                delta = tuple(-sh[i] for i in range(ibp_env.N_INDICES))
                # Try the actual call
                _, _, _, success = env.apply_action_resolved_target_only(
                    expr_t, subs, resolved_subs,
                    target, op, delta,
                    (0, 1, 1, 0, 1, 1, 1, 1),
                )
                if success:
                    return op, delta
    return None, None


def run_many(env, n_calls, expr_t, subs, resolved_subs, target, ibp_op, delta, target_sector):
    for _ in range(n_calls):
        env.apply_action_resolved_target_only(
            expr_t, subs, resolved_subs,
            target, ibp_op, delta, target_sector,
        )


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    N_SUBS = 200
    N_CALLS = 2000
    expr_t, subs, resolved_subs = grow_state(env, N_SUBS)
    target = random.Random(99).choice(list(subs.keys()))
    ibp_op, delta = find_working_action(env, target, expr_t, subs, resolved_subs)
    if ibp_op is None:
        # Fall back: pick a random target-sector integral
        for _ in range(50):
            target = random_tarsect_int(random.Random(_))
            ibp_op, delta = find_working_action(env, target, expr_t, subs, resolved_subs)
            if ibp_op is not None: break
        assert ibp_op is not None, 'Could not find working action'
    target_sector = (0, 1, 1, 0, 1, 1, 1, 1)

    print(f'Profiling {N_CALLS} calls to apply_action_resolved_target_only at |subs|={N_SUBS}')
    print('=' * 90)
    pr = cProfile.Profile()
    pr.enable()
    run_many(env, N_CALLS, expr_t, subs, resolved_subs, target, ibp_op, delta, target_sector)
    pr.disable()

    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    stats.print_stats(40)
    out = s.getvalue()
    # Print
    print(out)
