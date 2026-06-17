"""Measure get_raw_equation recompute cost vs a cache lookup, to decide whether
raw_eq_cache (50k full equations, ~0.5-1GB) can be replaced by recompute-on-
demand. Times N (op, seed) raw generations + reports avg equation size.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.topology import Topology      # noqa: E402
import sailir.ibp_env as ibp_env           # noqa: E402

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', 'topology_input', 'pentagonbox')
SEED = (1, 1, 1, 0, -2, 1, 1, 1, 0, 0, 0)   # memhog seed


def main():
    ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ibp_env.set_prime(1009)
    env = ibp_env.IBPEnvironment()
    ibp_t, li_t, shifts = env.ibp_t, env.li_t, env.shifts
    N = ibp_env.N_INDICES

    # build a batch of (op, seed) pairs around a few seeds (like one step does)
    pairs = []
    seeds = [SEED]
    for s in seeds:
        for off in range(-3, 4):
            base = tuple(s[i] + (off if i == 8 else 0) for i in range(N))
            for op, shift_list in shifts.items():
                for shift in shift_list:
                    seed = tuple(base[i] - shift[i] for i in range(N))
                    pairs.append((op, seed))
    print(f"{len(pairs)} (op, seed) pairs\n", flush=True)

    # how much of a raw is sub-weight? (start_w12 = the seed's weight)
    sw = ibp_env.weight(SEED)
    start_w12 = (sw[0], sw[1])

    def subweight(i):
        ww = ibp_env.weight(i)
        return (ww[0], ww[1]) < start_w12

    # cold recompute (no cache) — time get_raw_equation + sub-weight census
    t = time.time()
    sizes = []
    n_terms = n_sub = 0
    for op, seed in pairs:
        raw = ibp_env.get_raw_equation(ibp_t, li_t, op, seed)
        sizes.append(len(raw))
        for k in raw:
            n_terms += 1
            if subweight(k):
                n_sub += 1
    dt = time.time() - t
    per = dt / len(pairs) * 1e6
    avg_sz = sum(sizes) / max(len(sizes), 1)
    print(f"recompute get_raw_equation: {per:.1f} us/raw  ({dt*1000:.0f} ms for "
          f"{len(pairs)} raws), avg {avg_sz:.0f} terms/raw", flush=True)
    print(f"start_w12 (seed weight) = {start_w12}")
    print(f"SUB-WEIGHT terms in the raws: {n_sub:,} / {n_terms:,} "
          f"({100*n_sub/max(n_terms,1):.1f}%)  <-- what the cache is full of\n", flush=True)

    # warm (cached) lookup for comparison
    cache = {}
    for op, seed in pairs:
        cache[(op, seed)] = ibp_env.get_raw_equation(ibp_t, li_t, op, seed)
    t = time.time()
    for _ in range(5):
        for op, seed in pairs:
            _ = cache[(op, seed)]
    dt2 = (time.time() - t) / 5
    print(f"cache lookup:               {dt2/len(pairs)*1e6:.3f} us/raw", flush=True)
    print(f"\nrecompute is ~{per/(dt2/len(pairs)*1e6):.0f}x slower than a cache hit.")
    print(f"If a deep step touches ~5000 raws: recompute ~= "
          f"{5000*per/1000:.0f} ms/step extra (vs ~0 cached).")


if __name__ == '__main__':
    main()
