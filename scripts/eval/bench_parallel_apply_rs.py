"""Gating experiment for shared-memory process-pool parallelism.

Loads a DEEP resolved_subs, packs it into shared memory, then runs the rs-bound
kernel (apply_resolved_subs over many equations) SERIAL vs a persistent
ProcessPool whose workers attach to the shared arrays once (zero-copy, no pickle
of rs per task). Measures speedup and verifies parallel==serial.

This is the kernel enumerate Phase-1a spends its time in (apply_resolved_subs on
raws). If pooling it over shared-memory rs gives a real speedup, the full
enumerate/apply pool is worth building; if the fork/IPC overhead eats it, we
learn that here for the cost of one short run instead of a 5h reduction.

Usage: python bench_parallel_apply_rs.py <ckpt.pkl> <n_workers> <n_eqs> <terms> <sub_frac>
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.packed_eq import IntegralRegistry, PackedEq, _combine  # noqa: E402
from sailir.packed_resolved_subs import PackedResolvedSubs         # noqa: E402

PRIME = 1009

# ---- the kernel: apply_resolved_subs over a CSR-packed rs (one flat pass) ----
def apply_rs_csr(eq_ids, eq_coeffs, prs, prime):
    """Substitute every sub-key present in eq with its packed solution. rs
    solutions are flat (no sub-keys) so one pass suffices. Returns combined
    (ids, coeffs)."""
    out_ids = [eq_ids]
    out_co = [eq_coeffs.astype(np.int64)]
    keep = np.ones(len(eq_ids), dtype=bool)
    sids = prs.sub_ids
    for j in range(len(eq_ids)):
        sid = int(eq_ids[j])
        pos = int(np.searchsorted(sids, sid))
        if pos < len(sids) and sids[pos] == sid:       # sub-key present
            keep[j] = False
            a = int(prs.offsets[pos]); b = int(prs.offsets[pos + 1])
            c = int(eq_coeffs[j])
            out_ids.append(prs.flat_ids[a:b])
            out_co.append((c * prs.flat_coeffs[a:b].astype(np.int64)) % prime)
    if keep.all():
        return eq_ids, eq_coeffs
    out_ids[0] = eq_ids[keep]
    out_co[0] = eq_coeffs[keep].astype(np.int64)
    all_ids = np.concatenate(out_ids).astype(np.int64)
    all_co = np.concatenate(out_co)
    res = _combine(all_ids, all_co, prime)
    return res.ids, res.coeffs


# ---- worker: attach to shared rs once, process a chunk of equations ----
_W = {}
def _init(desc):
    _W['prs'] = PackedResolvedSubs.attach(desc)

def _work(chunk):
    prs = _W['prs']
    out = []
    for ids, cos in chunk:
        rids, rcos = apply_rs_csr(ids, cos, prs, PRIME)
        out.append(int(rids.sum()) ^ int(rcos.sum()))   # cheap checksum
    return out


def load_ckpt(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        if isinstance(d, dict) and d.get('_streamed'):
            n = d['n_states']; d = dict(d)
            d['beam'] = [pickle.load(f) for _ in range(n)]
    return d


def main():
    path = sys.argv[1]
    nws = [int(x) for x in sys.argv[2].split(',')]   # sweep worker counts
    n_eqs = int(sys.argv[3])
    terms = int(sys.argv[4]); sub_frac = float(sys.argv[5])

    print(f"loading {path} ...", flush=True)
    ck = load_ckpt(path)
    beam = ck['beam'] if isinstance(ck, dict) else ck
    rs = {}
    for st in beam:
        r = st.get('resolved_subs') if isinstance(st, dict) else getattr(st, 'resolved_subs', None)
        if r and len(r) > len(rs):
            rs = r
    print(f"resolved_subs: {len(rs):,} subs, {sum(len(v) for v in rs.values()):,} terms", flush=True)

    reg = IntegralRegistry()
    prs = PackedResolvedSubs.from_dict(rs, reg)
    n_sub = len(prs.sub_ids)
    n_int = len(reg)
    print(f"packed: {prs.n_terms:,} terms, {prs.flat_ids.nbytes/1e6:.1f} MB flat_ids; "
          f"registry {n_int:,} integrals", flush=True)

    # synthetic equations: `terms` ids each, `sub_frac` of them real sub-keys
    rng = np.random.default_rng(0)
    n_subkeys = max(1, int(terms * sub_frac))
    eqs = []
    for _ in range(n_eqs):
        sk = prs.sub_ids[rng.integers(0, n_sub, n_subkeys)]
        other = rng.integers(0, n_int, terms - n_subkeys).astype(np.int32)
        ids = np.unique(np.concatenate([sk, other]).astype(np.int32))
        cos = rng.integers(1, PRIME, len(ids)).astype(np.int16)
        eqs.append((ids, cos))
    print(f"{n_eqs} synthetic eqs, {terms} terms each, {n_subkeys} sub-keys each\n", flush=True)

    # serial
    t = time.time()
    serial = []
    for ids, cos in eqs:
        rids, rcos = apply_rs_csr(ids, cos, prs, PRIME)
        serial.append(int(rids.sum()) ^ int(rcos.sum()))
    t_ser = time.time() - t
    print(f"SERIAL:           {t_ser:.2f}s", flush=True)

    # parallel: persistent pool, workers attach to shared memory (sweep nw)
    desc = prs.to_shared(tag=f"prsb{os.getpid()}")
    for nw in nws:
        chunks = [eqs[i::nw] for i in range(nw)]
        t = time.time()
        with Pool(nw, initializer=_init, initargs=(desc,)) as pool:
            results = pool.map(_work, chunks)
        t_par = time.time() - t
        par = [None] * n_eqs
        for w in range(nw):
            for k, v in enumerate(results[w]):
                par[w + k * nw] = v
        ok = (par == serial)
        sp = t_ser / max(t_par, 1e-9)
        print(f"PARALLEL {nw:>2}w:  {t_par:6.2f}s   speedup={sp:5.2f}x   "
              f"eff={100*sp/nw:3.0f}%   correctness={'PASS' if ok else 'FAIL'}",
              flush=True)
    prs.close_shared(unlink=True)


if __name__ == '__main__':
    main()
