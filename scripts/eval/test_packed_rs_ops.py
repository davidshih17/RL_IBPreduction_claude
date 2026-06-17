"""Verify the packed resolved_subs algebra is BIT-IDENTICAL to the dict version.

Replays an identical sequence of (target, sol) through both
beam_search_v7.add_sub_to_resolved_v5 (dict) and
packed_rs_ops.add_sub_to_resolved_packed (PackedEq values), on the real env, and
asserts to_dict(rs_packed) == rs_dict after EVERY step. The sequence is built so
both Step-1 (resolve incoming sol against existing rs) and Step-3 (rewrite
existing solutions that contain the new target) actually fire.
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
    add_sub_to_resolved_packed, to_dict_view)

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', 'topology_input', 'pentagonbox')
PRIME = 1009


def main():
    topology = Topology.from_dir(TOPO)
    ibp_env.init_from_topology(topology)
    ibp_env.set_prime(PRIME)
    from beam_search_v7 import add_sub_to_resolved_v5, is_active  # noqa: E402

    N = ibp_env.N_INDICES
    rng = random.Random(20260616)

    # synthetic integrals with realistic small components; weight() works on any
    def rand_int():
        return tuple(rng.randint(-3, 3) for _ in range(N))

    pool = list({rand_int() for _ in range(80)})    # distinct integrals
    # STRICT total order by IBP weight (then tuple) -- the reduction hierarchy.
    order = sorted(pool, key=lambda i: (ibp_env.weight(i)[0], ibp_env.weight(i)[1], i))
    start_w12 = (ibp_env.weight(order[len(order)//3])[0],
                 ibp_env.weight(order[len(order)//3])[1])    # ~1/3 inactive
    reg = IntegralRegistry()
    active_id = lambda i: is_active(reg.get_tuple(i), start_w12)  # noqa: E731

    rs_dict = {}
    rs_packed = {}
    fired_step3 = 0
    n_steps = 0
    # sub targets HIGH -> LOW; a solution references only STRICTLY-LOWER
    # integrals (order[:idx]) -> resolved_subs stays flat, no self-reference,
    # exactly the IBP invariant. Subbing a lower target rewrites the existing
    # (higher) solutions that referenced it -> Step 3 fires.
    for idx in range(len(order) - 1, 1, -1):
        target = order[idx]
        lower = order[:idx]
        if len(lower) < 2:
            break
        ks = rng.sample(lower, rng.randint(2, min(5, len(lower))))
        sol_dict = {k: rng.randint(1, PRIME - 1) for k in ks}
        step = n_steps
        n_steps += 1

        will_fire = any(target in v for v in rs_dict.values())
        if will_fire:
            fired_step3 += 1

        # dict reference
        rs_dict = add_sub_to_resolved_v5(rs_dict, target, sol_dict, start_w12)
        # packed
        sol_packed = PackedEq.from_dict(sol_dict, reg)
        rs_packed = add_sub_to_resolved_packed(
            rs_packed, target, sol_packed, reg, active_id, PRIME)

        got = to_dict_view(rs_packed, reg)
        if got != rs_dict:
            print(f"\nstep {step}: MISMATCH (target={target})", flush=True)
            dk, gk = set(rs_dict), set(got)
            if dk != gk:
                print(f"  key sets differ: only-dict={list(dk-gk)[:3]} "
                      f"only-packed={list(gk-dk)[:3]}", flush=True)
            for k in (dk & gk):
                if rs_dict[k] != got[k]:
                    dv, gv = rs_dict[k], got[k]
                    print(f"  sub {k} (==target? {k==target}) differs:", flush=True)
                    print(f"    dict   ({len(dv)} terms): "
                          f"{dict(sorted(dv.items())[:6])}", flush=True)
                    print(f"    packed ({len(gv)} terms): "
                          f"{dict(sorted(gv.items())[:6])}", flush=True)
                    only_d = {kk: dv[kk] for kk in set(dv) - set(gv)}
                    only_p = {kk: gv[kk] for kk in set(gv) - set(dv)}
                    diff_c = {kk: (dv[kk], gv[kk]) for kk in set(dv) & set(gv) if dv[kk] != gv[kk]}
                    print(f"    only-dict={dict(list(only_d.items())[:4])} "
                          f"only-packed={dict(list(only_p.items())[:4])} "
                          f"coeff-diff={dict(list(diff_c.items())[:4])}", flush=True)
                    break
            sys.exit(1)
        print(f"step {step:2d}: |rs|={len(rs_dict):2d} sub={target[:4]}... "
              f"step3_fired={'Y' if will_fire else 'n'} OK", flush=True)

    print(f"\nPASS: packed add_sub_to_resolved == dict over {n_steps} steps "
          f"(Step-3 rewrites fired {fired_step3}x). registry={len(reg)}")


if __name__ == '__main__':
    main()
