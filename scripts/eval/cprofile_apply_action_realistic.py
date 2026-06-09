"""
Realistic-mix cProfile: many DIFFERENT (target, op, delta) tuples,
mimicking what a worker actually sees in one batch.
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


def build_realistic_batch(env, expr_t, subs, resolved_subs, n_actions, seed=99):
    """Build n_actions distinct (target, ibp_op, delta) tuples, similar to
    what apply_actions_worker iterates over in production."""
    rng = random.Random(seed)
    target_sector = (0, 1, 1, 0, 1, 1, 1, 1)
    batch = []
    attempts = 0
    while len(batch) < n_actions and attempts < n_actions * 20:
        attempts += 1
        target = random_tarsect_int(rng)
        op = rng.choice(list(env.shifts.keys()))
        sh = rng.choice(env.shifts[op])
        seed_t = tuple(target[i] - sh[i] for i in range(ibp_env.N_INDICES))
        raw = get_raw_equation(env.ibp_t, env.li_t, op, seed_t)
        if target not in raw or raw[target] == 0:
            continue
        delta = tuple(-sh[i] for i in range(ibp_env.N_INDICES))
        # Verify the call would succeed
        _, _, _, ok = env.apply_action_resolved_target_only(
            expr_t, subs, resolved_subs, target, op, delta, target_sector,
        )
        if ok:
            batch.append((target, op, delta))
    return batch, target_sector


def run_batch(env, batch, expr_t, subs, resolved_subs, target_sector):
    for target, op, delta in batch:
        env.apply_action_resolved_target_only(
            expr_t, subs, resolved_subs, target, op, delta, target_sector,
        )


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    N_SUBS = 200
    N_ACTIONS = 200  # ~one worker's batch
    expr_t, subs, resolved_subs = grow_state(env, N_SUBS)
    print(f'Building realistic batch of {N_ACTIONS} distinct actions at |subs|={N_SUBS}...')
    batch, target_sector = build_realistic_batch(env, expr_t, subs, resolved_subs, N_ACTIONS)
    print(f'Built {len(batch)} actions; |env._raw_eq_cache| after batch-building = {len(env._raw_eq_cache)}')

    # Measure cache hit rate over a fresh batch (clear the cache built during
    # batch construction)
    env._raw_eq_cache.clear()

    # Count cache lookups
    hits_before = misses_before = 0
    original = env.get_raw_equation_cached
    def counted(self, ibp_op, seed):
        key = (ibp_op, seed)
        global misses_before, hits_before
        if key in self._raw_eq_cache:
            hits_before += 1
        else:
            misses_before += 1
        return original(ibp_op, seed)
    env.get_raw_equation_cached = lambda op, s: counted(env, op, s)

    run_batch(env, batch, expr_t, subs, resolved_subs, target_sector)
    print(f'Single-pass: hits={hits_before}  misses={misses_before}  (hit rate {100*hits_before/(hits_before+misses_before):.1f}%)')

    # Second pass — cache fully populated
    hits_before = misses_before = 0
    run_batch(env, batch, expr_t, subs, resolved_subs, target_sector)
    print(f'Re-run pass: hits={hits_before}  misses={misses_before}  (hit rate {100*hits_before/(hits_before+misses_before):.1f}%)')

    # Restore + cProfile a cold-cache pass (realistic worker startup)
    env.get_raw_equation_cached = original.__get__(env, type(env))
    env._raw_eq_cache.clear()

    print()
    print('=== cProfile: COLD-cache pass over realistic batch ===')
    pr = cProfile.Profile()
    pr.enable()
    run_batch(env, batch, expr_t, subs, resolved_subs, target_sector)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
    print(s.getvalue())

    print('=== cProfile: WARM-cache pass over the same batch ===')
    pr = cProfile.Profile()
    pr.enable()
    run_batch(env, batch, expr_t, subs, resolved_subs, target_sector)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
    print(s.getvalue())
