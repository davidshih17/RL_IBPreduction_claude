"""Confirm the rewritten get_raw_equation(min_w12) (weight computed inline from
seed+shift, tuple built only for kept terms) is BIT-IDENTICAL to the reference:
the full equation filtered to (w1,w2) >= min_w12. Over many (op,seed) raws.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sailir.topology import Topology      # noqa: E402
import sailir.ibp_env as ibp_env           # noqa: E402

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'topology_input', 'pentagonbox')
SEED = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)


def main():
    ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ibp_env.set_prime(1009)
    env = ibp_env.IBPEnvironment()
    ibp_t, li_t = env.ibp_t, env.li_t
    N = ibp_env.N_INDICES
    sw = ibp_env.weight(SEED)
    w12 = (sw[0], sw[1])

    bad = n = 0
    for off in range(-5, 6):
        base = tuple(SEED[i] + (off if i == 8 else 0) for i in range(N))
        for op, shift_list in env.shifts.items():
            for shift in shift_list:
                seed = tuple(base[i] - shift[i] for i in range(N))
                full = ibp_env.get_raw_equation(ibp_t, li_t, op, seed)
                ref = {k: v for k, v in full.items()
                       if (ibp_env.weight(k)[0], ibp_env.weight(k)[1]) >= w12}
                got = ibp_env.get_raw_equation(ibp_t, li_t, op, seed, min_w12=w12)
                n += 1
                if got != ref:
                    bad += 1
                    if bad <= 3:
                        print(f"MISMATCH op={op} seed={seed}: "
                              f"ref={len(ref)} got={len(got)}", flush=True)
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: strip == full-then-filter over "
          f"{n} raws ({bad} mismatches), w12={w12}", flush=True)


if __name__ == '__main__':
    main()
