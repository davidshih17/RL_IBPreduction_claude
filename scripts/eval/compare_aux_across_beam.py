"""Measure how much aux_flat actually OVERLAPS across beam states, to decide
whether the per-state aux duplication is shareable or genuinely distinct.

Reports:
  - resolved_subs: keys shared across ALL states vs union (path divergence).
  - cu entries: how many raw-equation entries are identical across states
    (fully shared) vs total distinct -> the real dedup potential.
"""
import pickle
import sys


def main():
    path = sys.argv[1]
    print(f'loading {path} ...', flush=True)
    ck = pickle.load(open(path, 'rb'))
    beam = ck['beam']
    n = len(beam)
    print(f'beam states: {n}\n')

    # --- aux structure of state 0 ---
    aux0 = beam[0].get('aux_flat')
    print(f'aux_flat type: {type(aux0).__name__}, '
          f'len: {len(aux0) if hasattr(aux0,"__len__") else "?"}')
    for i, part in enumerate(aux0):
        ln = len(part) if hasattr(part, '__len__') else '?'
        print(f'  aux[{i}]: {type(part).__name__}, len={ln}')
    # cu is the list of substituted raw-equation dicts (the big part) — find it:
    # it is the part that is a list of dicts.
    cu_idx = None
    for i, part in enumerate(aux0):
        if isinstance(part, list) and part and isinstance(part[0], dict):
            cu_idx = i
            break
    print(f'\ncu = aux[{cu_idx}] (list of equation dicts)\n')

    # --- resolved_subs overlap ---
    rs_keysets = [set(beam[i].get('resolved_subs', {}).keys()) for i in range(n)]
    common = set.intersection(*rs_keysets) if rs_keysets else set()
    union = set.union(*rs_keysets) if rs_keysets else set()
    print('=== resolved_subs ===')
    print(f'  per-state sizes: {[len(k) for k in rs_keysets]}')
    print(f'  shared by ALL states: {len(common)}')
    print(f'  union (any state)   : {len(union)}')
    print(f'  divergent (union-common): {len(union)-len(common)}')

    # --- cu entry sharing ---
    # Hash each cu entry (raw equation dict) canonically; count multiplicity
    # across states.
    from collections import Counter
    entry_count = Counter()       # entry-hash -> # states containing it
    per_state_entries = []
    total_entries = 0
    for i in range(n):
        cu = beam[i]['aux_flat'][cu_idx]
        hs = set()
        for d in cu:
            h = hash(frozenset(d.items()))
            hs.add(h)
        per_state_entries.append(len(cu))
        total_entries += len(cu)
        for h in hs:
            entry_count[h] += 1
    distinct = len(entry_count)
    shared_all = sum(1 for h, c in entry_count.items() if c == n)
    print('\n=== cu entries (raw equations) ===')
    print(f'  per-state cu sizes: min={min(per_state_entries)} '
          f'max={max(per_state_entries)} mean={total_entries//n}')
    print(f'  total cu entries across beam : {total_entries}')
    print(f'  DISTINCT cu entries          : {distinct}')
    print(f'  shared by ALL {n} states     : {shared_all}')
    print(f'  dedup ratio (total/distinct) : {total_entries/distinct:.1f}x')
    print(f'\n  => if cu were interned/shared, ~{total_entries/distinct:.1f}x '
          f'fewer equation dicts in memory')


if __name__ == '__main__':
    main()
