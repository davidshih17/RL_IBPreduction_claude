"""Measure the deep cu (indirect_cache) size and the cost of packing its IDS +
union_bm to shared memory each step. enumerate Phase-1b (subsector path) needs
only cu ids (searchsorted membership) + union_bm (filter) -- NOT coeffs -- so the
shared cu for enumerate is ids+ubm only. This tells us whether per-step cu
re-pack is cheap enough for the enumerate pool.

Usage: python measure_cu_repack.py <ckpt.pkl>
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.packed_eq import IntegralRegistry             # noqa: E402


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
    # pick the state with the biggest cu
    best_cu = None
    for st in beam:
        aux = st.get('aux_flat') if isinstance(st, dict) else getattr(st, 'aux_flat', None)
        if aux is None:
            continue
        cu = aux[0]            # list of {integral:coeff} dicts (checkpoint form)
        if best_cu is None or len(cu) > len(best_cu):
            best_cu = cu
    if not best_cu:
        print("no cu found"); return
    n_entries = len(best_cu)
    n_terms = sum(len(c) for c in best_cu)
    print(f"deep cu: {n_entries:,} entries, {n_terms:,} total terms\n", flush=True)

    reg = IntegralRegistry()
    # warm registry
    for c in best_cu:
        for k in c:
            reg.get_id(k)

    REPS = 5
    t = time.time()
    for _ in range(REPS):
        offsets = np.empty(n_entries + 1, np.int64); offsets[0] = 0
        chunks = []
        for i, c in enumerate(best_cu):
            ids = np.fromiter((reg.get_id(k) for k in c), np.int32, len(c))
            ids.sort()
            chunks.append(ids)
            offsets[i + 1] = offsets[i] + len(c)
        flat_ids = np.concatenate(chunks) if chunks else np.empty(0, np.int32)
    t_build = (time.time() - t) / REPS

    from multiprocessing import shared_memory
    t = time.time()
    for j in range(REPS):
        shm = shared_memory.SharedMemory(create=True, size=max(flat_ids.nbytes, 1),
                                         name=f"cu{os.getpid()}_{j}")
        view = np.ndarray(flat_ids.shape, np.int32, buffer=shm.buf)
        view[:] = flat_ids[:]
        shm.close(); shm.unlink()
    t_shm = (time.time() - t) / REPS

    print(f"cu ids: {flat_ids.nbytes/1e6:.1f} MB", flush=True)
    print(f"build cu-ids CSR (warm reg): {t_build*1000:.0f} ms", flush=True)
    print(f"copy to shm:                 {t_shm*1000:.0f} ms", flush=True)
    print(f"TOTAL cu re-pack:            {(t_build+t_shm)*1000:.0f} ms", flush=True)
    print(f"\n+ rs re-pack 71ms = {(t_build+t_shm)*1000+71:.0f} ms/step shared-state cost "
          f"(~{100*((t_build+t_shm)+0.071)/4.5:.0f}% of a deep ~4.5s step)", flush=True)


if __name__ == '__main__':
    main()
