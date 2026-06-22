"""Classify the lower-weight terms in the cu (cached IBP equations) to decide if
the 3.6% is a bug or expected:

  - TARGET-SECTOR PASSENGER: all target propagators present, weight < start_w12.
    expr discards these; an IBP identity mod-lower-weight could strip them too,
    so their presence in the cu is at best wasted memory, at worst inconsistent
    with "work mod lower-weight everywhere".
  - SUB-SECTOR: a target propagator is pinched (integral[i] <= 0 where
    target_sector[i]==1). These define the sector-filter behaviour and CANNOT be
    naively stripped.

Usage: python investigate_cu_lowweight.py <ckpt.pkl>
"""
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.topology import Topology      # noqa: E402
import sailir.ibp_env as ibp_env           # noqa: E402

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
    ibp_env.init_from_topology(Topology.from_dir(TOPO))
    ck = load_ckpt(path)
    start_w12 = tuple(ck['start_w12'])
    target_sector = tuple(ck['target_sector'])
    nd = ibp_env.N_DENOMINATORS
    print(f"start_w12={start_w12}  target_sector={target_sector}  N_DEN={nd}\n", flush=True)

    def w(i):
        ww = ibp_env.weight(i)
        return (ww[0], ww[1])

    def in_target_sector(integral):
        for i in range(nd):
            if (1 if integral[i] > 0 else 0) != target_sector[i]:
                return False
        return True

    beam = ck['beam'] if isinstance(ck, dict) else ck
    n_low_target = n_low_sub = n_terms = n_low = 0
    ex_target = []
    ex_sub = []
    low_target_weights = Counter()
    for st in beam:
        aux = st.get('aux_flat') if isinstance(st, dict) else getattr(st, 'aux_flat', None)
        if aux is None:
            continue
        for cu_entry in aux[0]:
            for k in cu_entry:
                n_terms += 1
                if w(k) < start_w12:
                    n_low += 1
                    if in_target_sector(k):
                        n_low_target += 1
                        low_target_weights[w(k)] += 1
                        if len(ex_target) < 5: ex_target.append((k, w(k)))
                    else:
                        n_low_sub += 1
                        if len(ex_sub) < 5: ex_sub.append((k, w(k)))

    print(f"cu terms total: {n_terms:,}")
    print(f"  lower-weight total:        {n_low:,} ({100*n_low/max(n_terms,1):.2f}%)")
    print(f"    TARGET-SECTOR passenger: {n_low_target:,} "
          f"({100*n_low_target/max(n_low,1):.1f}% of lower-weight)  <-- strippable?")
    print(f"    SUB-SECTOR:              {n_low_sub:,} "
          f"({100*n_low_sub/max(n_low,1):.1f}% of lower-weight)  <-- needed for filter")
    print(f"\n  target-sector-passenger weights seen: {dict(low_target_weights.most_common(8))}")
    print("\n  examples TARGET-SECTOR passengers (all props present, low weight):")
    for k, ww in ex_target:
        print(f"    weight={ww}  {k}")
    print("  examples SUB-SECTOR (pinched propagator):")
    for k, ww in ex_sub:
        print(f"    weight={ww}  {k}")


if __name__ == '__main__':
    main()
