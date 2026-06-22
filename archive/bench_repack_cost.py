"""Measure the per-step cost of (re)packing resolved_subs to shared memory.

In the real loop resolved_subs changes every step (new sub + COW-rewrites), so
the shared-memory packed rs must be rebuilt each step before dispatching to the
pool. This times from_dict (build CSR) + to_shared (copy to shm) on the DEEP rs,
to compare against the per-step enumerate work (~4-5s at deep steps) the pool
would parallelize. If re-pack << enumerate, full re-pack each step is fine.

Usage: python bench_repack_cost.py <ckpt.pkl>
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.packed_eq import IntegralRegistry                  # noqa: E402
from sailir.packed_resolved_subs import PackedResolvedSubs     # noqa: E402


def load_ckpt(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        if isinstance(d, dict) and d.get('_streamed'):
            n = d['n_states']; d = dict(d)
            d['beam'] = [pickle.load(f) for _ in range(n)]
    return d


def main():
    path = sys.argv[1]
    ck = load_ckpt(path)
    beam = ck['beam'] if isinstance(ck, dict) else ck
    rs = {}
    for st in beam:
        r = st.get('resolved_subs') if isinstance(st, dict) else getattr(st, 'resolved_subs', None)
        if r and len(r) > len(rs):
            rs = r
    n_terms = sum(len(v) for v in rs.values())
    print(f"deep resolved_subs: {len(rs):,} subs, {n_terms:,} terms\n", flush=True)

    reg = IntegralRegistry()
    PackedResolvedSubs.from_dict(rs, reg)   # warm the registry (one-time)

    REPS = 5
    # from_dict (build CSR) with a WARM registry (steady-state: ids already interned)
    t = time.time()
    for _ in range(REPS):
        prs = PackedResolvedSubs.from_dict(rs, reg)
    t_pack = (time.time() - t) / REPS

    # to_shared (copy to shm) + close
    t = time.time()
    for i in range(REPS):
        desc = prs.to_shared(tag=f"rp{os.getpid()}_{i}")
        prs.close_shared(unlink=True)
    t_shm = (time.time() - t) / REPS

    total = t_pack + t_shm
    print(f"from_dict (build CSR, warm reg): {t_pack*1000:.0f} ms", flush=True)
    print(f"to_shared (copy to shm):         {t_shm*1000:.0f} ms", flush=True)
    print(f"TOTAL per-step re-pack:          {total*1000:.0f} ms", flush=True)
    print(f"\nvs a deep enumerate step ~4-5s -> re-pack overhead "
          f"~{100*total/4.5:.1f}% of one step", flush=True)


if __name__ == '__main__':
    main()
