"""
Profile the inner operations of apply_action_resolved_target_only at
realistic late-step state. Decomposes per-action cost into:
   1. get_raw_equation (cached)
   2. apply_resolved_subs(raw, resolved_subs)  → "cached" dict
   3. solve_ibp_for(cached, target)
   4. apply_substitution_target_only(expr_t, target, sol_target, target_sector)
   5. add_sub_to_resolved(resolved_subs, target, sol_target)

Plus the cost of preparing the call:
   - dict(state.expr) / dict(state.subs) at call entry
   - the "sol_target" subsetting

Then sums to per-step cost at typical (|subs|, |candidates|) settings.
"""
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime,
    get_raw_equation, apply_resolved_subs, solve_ibp_for,
    apply_substitution_target_only, add_sub_to_resolved,
    integral_in_exact_sector,

)


def random_tarsect_int(rng):
    """Sample a plausible target-sector pentagon-box integral."""
    out = []
    for i in range(ibp_env.N_INDICES):
        if i < 8:
            out.append(rng.randint(1, 3))
        else:
            out.append(rng.randint(-3, 0))
    return tuple(out)


def random_sol(rng, n_terms=15):
    sol = {}
    for _ in range(n_terms):
        sol[random_tarsect_int(rng)] = rng.randint(1, ibp_env.PRIME - 1)
    return sol


def grow_state(env, N, seed=42):
    """Build (expr_t, subs, resolved_subs) at ~|subs|=N by applying N
    random substitutions to an initially-trivial state."""
    rng = random.Random(seed)
    expr_t = random_sol(rng, n_terms=25)
    subs = {}
    resolved_subs = {}
    for _ in range(N):
        target = random_tarsect_int(rng)
        sol = random_sol(rng, n_terms=rng.randint(8, 22))
        subs[target] = sol
        resolved_subs = add_sub_to_resolved(resolved_subs, target, sol)
    # And one more "action target" + delta to use as the action
    new_target = random_tarsect_int(rng)
    return expr_t, subs, resolved_subs, new_target, rng


def time_phase(fn, label, n=200, **kwargs):
    """Execute `fn` n times, return mean ms."""
    t0 = time.perf_counter()
    last = None
    for _ in range(n):
        last = fn()
    dt = time.perf_counter() - t0
    return label, 1000 * dt / n, last


def bench_at(env, N_subs, n_trials=200):
    """Decompose apply_action_resolved_target_only at |subs|=N_subs."""
    expr_t, subs, resolved_subs, target, rng = grow_state(env, N_subs)
    target_sector = (0, 1, 1, 0, 1, 1, 1, 1)  # the (8,4)-style target sector

    # Pick an IBP op + delta that succeeds (target stays in raw)
    ibp_op = 0
    delta = (0,) * 11
    # Find one that works
    found = False
    for op in env.shifts:
        for sh in env.shifts[op]:
            seed = tuple(target[i] - sh[i] for i in range(ibp_env.N_INDICES))
            raw = get_raw_equation(env.ibp_t, env.li_t, op, seed)
            if target in raw and raw[target] != 0:
                ibp_op = op
                delta = tuple(-sh[i] for i in range(ibp_env.N_INDICES))
                found = True
                break
        if found: break
    if not found:
        print(f'  No succeeding (op, delta) found at N={N_subs}; skipping.')
        return

    # The seed actually used by apply_action_resolved_target_only:
    # seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
    seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))

    # Warmup raw eq cache
    for _ in range(3):
        get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
    # raw is now in env._raw_eq_cache (after cache lookup hot)

    # --- Phase A: get_raw_equation cached lookup ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        key = (ibp_op, seed)
        raw = env._raw_eq_cache.get(key)
        if raw is None:
            raw = get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
            env._raw_eq_cache[key] = raw
    t_get_raw = (time.perf_counter() - t0) / n_trials

    # --- Phase B: apply_resolved_subs(raw, resolved_subs) ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        cached = apply_resolved_subs(raw, resolved_subs)
    t_apply_rs = (time.perf_counter() - t0) / n_trials

    # --- Phase C: solve_ibp_for(cached, target) ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        sol = solve_ibp_for(cached, target)
    t_solve = (time.perf_counter() - t0) / n_trials

    if sol is None:
        print(f'  solve_ibp_for returned None at N={N_subs}; skipping.')
        return

    # --- Phase D: subset sol to target-sector ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        sol_target = {k: v for k, v in sol.items()
                      if integral_in_exact_sector(k, target_sector)}
    t_sol_target = (time.perf_counter() - t0) / n_trials

    # --- Phase E: dict(state.expr) + dict(state.subs) prep ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        et = dict(expr_t)
        ss = dict(subs)
    t_prep_dicts = (time.perf_counter() - t0) / n_trials

    # --- Phase F: apply_substitution_target_only ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        new_expr_t = apply_substitution_target_only(
            expr_t, target, sol_target, target_sector
        )
    t_apply_sub = (time.perf_counter() - t0) / n_trials

    # --- Phase G: add_sub_to_resolved ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        new_rs = add_sub_to_resolved(resolved_subs, target, sol_target)
    t_add_sub = (time.perf_counter() - t0) / n_trials

    # --- Total: full apply_action_resolved_target_only ---
    t0 = time.perf_counter()
    for _ in range(n_trials):
        env.apply_action_resolved_target_only(
            dict(expr_t), dict(subs), resolved_subs,
            target, ibp_op, delta, target_sector,
        )
    t_full = (time.perf_counter() - t0) / n_trials

    parts = [
        ('A. get_raw_equation (cache)', t_get_raw),
        ('B. apply_resolved_subs(raw,rs)', t_apply_rs),
        ('C. solve_ibp_for', t_solve),
        ('D. subset sol -> sol_target', t_sol_target),
        ('E. dict(expr)+dict(subs) prep', t_prep_dicts),
        ('F. apply_substitution_target_only', t_apply_sub),
        ('G. add_sub_to_resolved (post-COW)', t_add_sub),
    ]
    overhead = t_full - sum(p[1] for p in parts)
    parts.append(('-. residual / overhead', overhead))

    print(f'\n--- |subs| = {N_subs}  full call mean = {1000*t_full:.3f} ms ---')
    print(f'{"phase":<40} {"ms":>10}  {"%full":>8}')
    for label, t in parts:
        pct = 100 * t / t_full if t_full > 0 else 0
        print(f'{label:<40} {1000*t:>10.3f}  {pct:>7.1f}%')
    print(f'{"--- TOTAL (measured)":<40} {1000*t_full:>10.3f}')


if __name__ == '__main__':
    topology = Topology.from_dir(str(_HERE.parent.parent.parent / 'topology_input' / 'pentagonbox'))
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    env = IBPEnvironment()

    print('=' * 80)
    print('apply_action_resolved_target_only: per-component breakdown')
    print('(post-COW add_sub_to_resolved)')
    print('=' * 80)
    for N in [50, 100, 200, 300]:
        bench_at(env, N, n_trials=200)
    print('=' * 80)
