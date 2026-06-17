"""Verify PackedResolvedSubs: (1) dict->pack->dict round-trip is bit-identical,
(2) the packed arrays in SHARED MEMORY are read correctly from a separate worker
process (attach-by-name, not fork inheritance), (3) report dict vs packed size.

Loads a checkpoint (streamed or legacy), grabs the largest resolved_subs across
the beam. Usage: python verify_packed_rs_shared.py <ckpt.pkl>
"""
import os
import pickle
import sys
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.packed_eq import IntegralRegistry          # noqa: E402
from sailir.packed_resolved_subs import PackedResolvedSubs  # noqa: E402


def load_ckpt(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        if isinstance(d, dict) and d.get('_streamed'):
            n = d['n_states']
            d = dict(d)
            d['beam'] = [pickle.load(f) for _ in range(n)]
    return d


def biggest_resolved_subs(ckpt):
    beam = ckpt['beam'] if isinstance(ckpt, dict) else ckpt
    best = {}
    for st in beam:
        rs = (st.get('resolved_subs') if isinstance(st, dict)
              else getattr(st, 'resolved_subs', None))
        if rs and len(rs) > len(best):
            best = rs
    return best


def _worker(desc, queries, q):
    # Separate process: attach to shared memory by name and look up.
    prs = PackedResolvedSubs.attach(desc)
    out = {}
    for sid in queries:
        r = prs.get(sid)
        out[sid] = None if r is None else (r[0].tolist(), r[1].tolist())
    q.put(out)


def main():
    path = sys.argv[1]
    print(f"loading {path} ...", flush=True)
    ckpt = load_ckpt(path)
    rs = biggest_resolved_subs(ckpt)
    n_terms = sum(len(v) for v in rs.values())
    print(f"resolved_subs: {len(rs):,} subs, {n_terms:,} total terms\n", flush=True)
    if not rs:
        print("EMPTY resolved_subs in this checkpoint — pick a deeper one.")
        return

    reg = IntegralRegistry()
    prs = PackedResolvedSubs.from_dict(rs, reg)

    # (1) round-trip
    back = prs.to_dict(reg)
    ok_rt = (back == rs)
    print(f"(1) round-trip dict==pack->dict: {'PASS' if ok_rt else 'FAIL'}")
    if not ok_rt:
        # locate first mismatch
        for k in rs:
            if rs[k] != back.get(k):
                print(f"   first mismatch at sub {k[:4]}...: "
                      f"{len(rs[k])} vs {len(back.get(k, {}))} terms")
                break

    # (2) shared memory across a worker process
    desc = prs.to_shared(tag=f"prs{os.getpid()}")
    import random
    rng = random.Random(0)
    sample_subs = rng.sample(list(rs.keys()), min(200, len(rs)))
    query_ids = [reg.get_id(k) for k in sample_subs]
    q = Queue()
    p = Process(target=_worker, args=(desc, query_ids, q))
    p.start()
    got = q.get()
    p.join()
    mism = 0
    for k in sample_subs:
        sid = reg.get_id(k)
        exp_ids = sorted(reg.get_id(i) for i in rs[k])
        exp = {reg.get_id(i): rs[k][i] for i in rs[k]}
        g = got[sid]
        if g is None:
            mism += 1
            continue
        gids, gcos = g
        if gids != exp_ids or any(exp[i] != c for i, c in zip(gids, gcos)):
            mism += 1
    print(f"(2) shared-memory worker read {len(sample_subs)} subs: "
          f"{'PASS' if mism == 0 else f'FAIL ({mism} mismatches)'}")

    # (3) sizes
    packed_bytes = (prs.sub_ids.nbytes + prs.offsets.nbytes
                    + prs.flat_ids.nbytes + prs.flat_coeffs.nbytes)
    # dict estimate ~ 64B/entry (slot + int) + 104B/dict, tuple keys shared
    dict_est = n_terms * 64 + len(rs) * 232
    print(f"\n(3) size: dict ~{dict_est/1e6:.0f} MB  vs  "
          f"packed {packed_bytes/1e6:.1f} MB  "
          f"({dict_est/max(packed_bytes,1):.0f}x smaller)  "
          f"registry={len(reg):,} integrals")
    prs.close_shared(unlink=True)
    print("\nVERDICT:", "PASS" if (ok_rt and mism == 0) else "FAIL")


if __name__ == '__main__':
    main()
