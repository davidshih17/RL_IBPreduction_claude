"""Microbenchmark the strip implementation. 3 local variants of get_raw_equation:
  full          : no strip (min_w12=None)
  strip_weight  : OLD — calls weight(integral), which allocates an unused 11-tuple
  strip_inline  : NEW — computes (w0,w1) inline, no tuple alloc, no tuple compare
Isolates the per-raw cost so we can see exactly how wasteful the weight() call is.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sailir.topology import Topology      # noqa: E402
import sailir.ibp_env as ibp_env           # noqa: E402

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'topology_input', 'pentagonbox')
SEED = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)   # the 74 start integral
REPS = 20


def make_variants(ibp_t, li_t):
    N = ibp_env.N_INDICES
    eval_coeff = ibp_env.eval_coeff
    weight = ibp_env.weight
    n_ibp = len(ibp_t)

    def _template(ibp_op):
        return (li_t.get(ibp_op - n_ibp, []) if ibp_op >= n_ibp
                else ibp_t.get(ibp_op, []))

    def full(ibp_op, seed):
        eq = {}
        for shift, coeff_str in _template(ibp_op):
            c = eval_coeff(coeff_str, seed)
            if c != 0:
                integral = tuple(seed[i] + shift[i] for i in range(N))
                eq[integral] = c
        return eq

    def strip_weight(ibp_op, seed, min_w12):
        eq = {}
        for shift, coeff_str in _template(ibp_op):
            c = eval_coeff(coeff_str, seed)
            if c != 0:
                integral = tuple(seed[i] + shift[i] for i in range(N))
                w = weight(integral)
                if (w[0], w[1]) < min_w12:
                    continue
                eq[integral] = c
        return eq

    def strip_inline(ibp_op, seed, min_w12):
        m0, m1 = min_w12
        eq = {}
        for shift, coeff_str in _template(ibp_op):
            c = eval_coeff(coeff_str, seed)
            if c != 0:
                integral = tuple(seed[i] + shift[i] for i in range(N))
                w0 = 0
                w1 = 0
                for x in integral:
                    if x > 0:
                        w0 += x
                    elif x < 0:
                        w1 -= x
                if w0 < m0 or (w0 == m0 and w1 < m1):
                    continue
                eq[integral] = c
        return eq

    def strip_preeval(ibp_op, seed, min_w12):
        # strip BEFORE eval_coeff: integral weight needs only seed+shift, so we
        # decide to drop a term WITHOUT paying the expensive eval_coeff for the
        # 28% we throw away. Stripping now does LESS work than full, not more.
        m0, m1 = min_w12
        eq = {}
        for shift, coeff_str in _template(ibp_op):
            integral = tuple(seed[i] + shift[i] for i in range(N))
            w0 = 0
            w1 = 0
            for x in integral:
                if x > 0:
                    w0 += x
                elif x < 0:
                    w1 -= x
            if w0 < m0 or (w0 == m0 and w1 < m1):
                continue
            c = eval_coeff(coeff_str, seed)
            if c != 0:
                eq[integral] = c
        return eq

    return full, strip_weight, strip_inline, strip_preeval


def main():
    ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ibp_env.set_prime(1009)
    env = ibp_env.IBPEnvironment()
    ibp_t, li_t = env.ibp_t, env.li_t
    N = ibp_env.N_INDICES
    pairs = []
    for off in range(-4, 5):
        base = tuple(SEED[i] + (off if i == 8 else 0) for i in range(N))
        for op, shift_list in env.shifts.items():
            for shift in shift_list:
                pairs.append((op, tuple(base[i] - shift[i] for i in range(N))))
    sw = ibp_env.weight(SEED)
    start_w12 = (sw[0], sw[1])
    print(f"{len(pairs)} (op,seed) pairs x {REPS} reps,  start_w12={start_w12}\n", flush=True)

    full, strip_weight, strip_inline, strip_preeval = make_variants(ibp_t, li_t)

    # correctness: every strip variant must produce IDENTICAL dicts
    bad = 0
    for op, seed in pairs:
        a = strip_weight(op, seed, start_w12)
        if (a != strip_inline(op, seed, start_w12)
                or a != strip_preeval(op, seed, start_w12)):
            bad += 1
    print(f"strip_weight == strip_inline == strip_preeval on all {len(pairs)} "
          f"raws: {bad == 0} ({bad} mismatches)\n", flush=True)

    def timeit(fn, stripped):
        t = time.time()
        for _ in range(REPS):
            for op, seed in pairs:
                fn(op, seed, start_w12) if stripped else fn(op, seed)
        return time.time() - t

    npairs = len(pairs) * REPS
    # warm
    for _ in range(2):
        timeit(full, False); timeit(strip_weight, True)
        timeit(strip_inline, True); timeit(strip_preeval, True)
    d_full = timeit(full, False)
    d_w = timeit(strip_weight, True)
    d_i = timeit(strip_inline, True)
    d_p = timeit(strip_preeval, True)
    print(f"full         : {d_full/npairs*1e6:6.2f} us/raw  (baseline)", flush=True)
    print(f"strip_weight : {d_w/npairs*1e6:6.2f} us/raw  "
          f"({(d_w-d_full)/d_full*100:+.0f}% vs full)   <- OLD (weight() tuple)", flush=True)
    print(f"strip_inline : {d_i/npairs*1e6:6.2f} us/raw  "
          f"({(d_i-d_full)/d_full*100:+.0f}% vs full)   <- inline w0/w1", flush=True)
    print(f"strip_preeval: {d_p/npairs*1e6:6.2f} us/raw  "
          f"({(d_p-d_full)/d_full*100:+.0f}% vs full)   <- strip BEFORE eval_coeff", flush=True)


if __name__ == '__main__':
    main()
