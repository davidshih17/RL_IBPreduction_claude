"""Verify no PASSENGER (weight < start_w12) integrals live in the active beam
state. is_active(I) = (w1,w2) >= start_w12; passengers are supposed to be
stripped from resolved_subs values and discarded from expr (recovered by
replay). Scans every state's expr + resolved_subs (keys AND value-keys) and
reports any integral whose weight is below start_w12.

cu (aux_flat) is NOT scanned: it holds raw IBP *equations* that relate a target
to lower integrals, so it legitimately contains below-threshold integrals.
sub_accum is reported separately: it's the sub-sector passenger accumulator
(Option F), so below-threshold there is by design.

Usage: python verify_no_passengers.py <ckpt.pkl>
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from sailir.topology import Topology          # noqa: E402
import sailir.ibp_env as ibp_env               # noqa: E402

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', 'topology_input', 'pentagonbox')


def load_ckpt(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        if isinstance(d, dict) and d.get('_streamed'):
            n = d['n_states']; d = dict(d)
            d['beam'] = [pickle.load(f) for _ in range(n)]
    return d


def main():
    path = sys.argv[1]
    Topology.from_dir(TOPO) and ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ck = load_ckpt(path)
    start_w12 = tuple(ck['start_w12']) if isinstance(ck, dict) else None
    beam = ck['beam'] if isinstance(ck, dict) else ck
    print(f"start_w12 = {start_w12} ; active = weight >= start_w12\n", flush=True)

    def w(i):
        ww = ibp_env.weight(i)
        return (ww[0], ww[1])

    def passenger(i):
        return w(i) < start_w12

    from collections import defaultdict
    tot = defaultdict(int)
    low = defaultdict(int)
    examples = defaultdict(list)

    def scan(where, integ):
        tot[where] += 1
        if passenger(integ):
            low[where] += 1
            if len(examples[where]) < 4:
                examples[where].append((integ, w(integ)))

    for st in beam:
        get = (lambda k: st.get(k)) if isinstance(st, dict) else (lambda k: getattr(st, k, None))
        for k in (get('expr') or {}):
            scan('expr', k)
        rs = get('resolved_subs') or {}
        for k, v in rs.items():
            scan('rs_key', k)
            for kk in v:
                scan('rs_value', kk)
        for k in (get('sub_accum') or {}):
            scan('sub_accum', k)
        aux = get('aux_flat')
        if aux is not None:
            for cu_entry in aux[0]:           # cu: list of {integral:coeff}
                for k in cu_entry:
                    scan('cu', k)

    print("COMPREHENSIVE lower-weight (weight < start_w12) scan across ALL structures:\n")
    for where in ('expr', 'rs_key', 'rs_value', 'sub_accum', 'cu'):
        if tot[where]:
            pct = 100 * low[where] / tot[where]
            print(f"  {where:10s}: {low[where]:>12,} / {tot[where]:>12,} "
                  f"lower-weight ({pct:.1f}%)")
    print()
    for where in ('expr', 'rs_key', 'rs_value', 'sub_accum', 'cu'):
        for integ, ww in examples[where]:
            print(f"  [{where}] weight={ww} < {start_w12}  {integ}")
    any_low = any(low[w_] for w_ in low)
    print(f"\nVERDICT: lower-weight integrals found ANYWHERE: "
          f"{'YES (see above)' if any_low else 'NONE'}")


if __name__ == '__main__':
    main()
