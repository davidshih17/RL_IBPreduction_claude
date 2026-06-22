"""Verify apply_resolved_subs_dict_x_packed (dict raw x packed rs -> dict) is
BIT-IDENTICAL to ibp_env.apply_resolved_subs(dict raw, dict rs).

Builds a realistic resolved_subs (same construction as test_packed_rs_ops.py),
then for many random expr dicts that DO contain sub-keys, asserts the packed
helper's output dict equals the dict-version output, after every rs step.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sailir.topology import Topology                    # noqa: E402
import sailir.ibp_env as ibp_env                         # noqa: E402
from sailir.packed_eq import IntegralRegistry, PackedEq  # noqa: E402
from sailir.packed_rs_ops import (                       # noqa: E402
    add_sub_to_resolved_packed, apply_resolved_subs_dict_x_packed)

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', 'topology_input', 'pentagonbox')
PRIME = 1009


def main():
    ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ibp_env.set_prime(PRIME)
    from beam_search_v7 import add_sub_to_resolved_v5, is_active  # noqa: E402

    N = ibp_env.N_INDICES
    rng = random.Random(20260617)

    def rand_int():
        return tuple(rng.randint(-3, 3) for _ in range(N))

    pool = list({rand_int() for _ in range(80)})
    order = sorted(pool, key=lambda i: (ibp_env.weight(i)[0], ibp_env.weight(i)[1], i))
    start_w12 = (ibp_env.weight(order[len(order)//3])[0],
                 ibp_env.weight(order[len(order)//3])[1])
    reg = IntegralRegistry()
    active_id = lambda i: is_active(reg.get_tuple(i), start_w12)  # noqa: E731

    rs_dict = {}
    rs_packed = {}
    n_steps = 0
    total_exprs = 0
    n_substituted = 0
    for idx in range(len(order) - 1, 1, -1):
        target = order[idx]
        lower = order[:idx]
        if len(lower) < 2:
            break
        ks = rng.sample(lower, rng.randint(2, min(5, len(lower))))
        sol_dict = {k: rng.randint(1, PRIME - 1) for k in ks}
        rs_dict = add_sub_to_resolved_v5(rs_dict, target, sol_dict, start_w12)
        rs_packed = add_sub_to_resolved_packed(
            rs_packed, target, PackedEq.from_dict(sol_dict, reg), reg, active_id, PRIME)
        n_steps += 1

        # build random exprs that DELIBERATELY include sub-keys so substitution fires
        sub_keys = list(rs_dict.keys())
        for _ in range(20):
            expr = {}
            if sub_keys:
                for k in rng.sample(sub_keys, rng.randint(1, min(4, len(sub_keys)))):
                    expr[k] = rng.randint(1, PRIME - 1)
            for _ in range(rng.randint(0, 5)):     # plus some non-sub integrals
                expr[rand_int()] = rng.randint(1, PRIME - 1)
            if not expr:
                continue
            total_exprs += 1
            ref = ibp_env.apply_resolved_subs(expr, rs_dict)
            got = apply_resolved_subs_dict_x_packed(expr, rs_packed, reg, PRIME)
            if any(k in rs_dict for k in expr):
                n_substituted += 1
            if ref != got:
                print(f"\nstep {n_steps}: MISMATCH", flush=True)
                dk, gk = set(ref), set(got)
                print(f"  only-ref={dict(list({k: ref[k] for k in dk-gk}.items())[:4])}")
                print(f"  only-got={dict(list({k: got[k] for k in gk-dk}.items())[:4])}")
                cd = {k: (ref[k], got[k]) for k in dk & gk if ref[k] != got[k]}
                print(f"  coeff-diff={dict(list(cd.items())[:4])}")
                sys.exit(1)

    print(f"\nPASS: apply_resolved_subs_dict_x_packed == dict version over "
          f"{total_exprs} exprs across {n_steps} rs steps "
          f"({n_substituted} actually substituted). registry={len(reg)}")


if __name__ == '__main__':
    main()
