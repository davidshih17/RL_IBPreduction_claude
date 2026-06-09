"""How do the 40 beam states' RS actually differ?
- Are RS keys the same across states? (= same set of substituted targets)
- For shared keys, do values differ?
- What % of (key, value) pairs are shared between state 0 and state i?

Also: is there a small "spine" of shared substitutions and a small "tail" of
per-state differences? Or are they completely different RS dicts?
"""
import pickle, sys
ckpt = sys.argv[1]
with open(ckpt, 'rb') as f:
    d = pickle.load(f)
beam = d['beam']
print(f'step={d["step"]} n_beam={len(beam)}')

# State 0 as reference
rs0 = beam[0]['resolved_subs']
keys0 = set(rs0.keys())
print(f'\nstate 0 RS: {len(rs0)} entries')

key_overlap_counts = []
value_match_counts = []
n_print = 5
for i in range(1, len(beam)):
    rsi = beam[i]['resolved_subs']
    keysi = set(rsi.keys())
    shared_keys = keys0 & keysi
    same_keys = (keys0 == keysi)
    # value comparison on shared keys
    n_val_eq = sum(1 for k in shared_keys if rs0[k] == rsi[k])
    if i <= n_print:
        print(f'  state {i}: |RS|={len(rsi)}  shared_keys={len(shared_keys)}  '
              f'same_keyset={same_keys}  values_eq_on_shared={n_val_eq}/{len(shared_keys)}')
    key_overlap_counts.append(len(shared_keys))
    value_match_counts.append(n_val_eq)

print(f'\nSummary across all 39 vs state 0:')
print(f'  shared keys (state i ∩ state 0): min={min(key_overlap_counts)} '
      f'max={max(key_overlap_counts)} all_same={len(set(key_overlap_counts))==1}')
print(f'  values equal on shared keys: '
      f'min={min(value_match_counts)} max={max(value_match_counts)} '
      f'all_same={len(set(value_match_counts))==1}')

# Pairwise key insertion order — are keys in same order?
import itertools
keys0_list = list(rs0.keys())
n_same_order = 0
for i in range(1, len(beam)):
    rsi = beam[i]['resolved_subs']
    if list(rsi.keys()) == keys0_list:
        n_same_order += 1
print(f'  states with SAME RS key insertion order as state 0: '
      f'{n_same_order}/{len(beam)-1}')

# Show the first few RS keys with their state-0 values
print(f'\nFirst 10 RS keys (state 0) and where they entered:')
for k in keys0_list[:10]:
    v = rs0[k]
    print(f'  {k} -> {len(v)} terms')

# Compare value sizes across beam for the FIRST RS key
print(f'\nFor state-0 RS key 0 = {keys0_list[0]}:')
v0 = rs0[keys0_list[0]]
for i in range(min(8, len(beam))):
    rsi = beam[i]['resolved_subs']
    if keys0_list[0] in rsi:
        same = (rsi[keys0_list[0]] == v0)
        print(f'  state {i}: present, equal_to_state0={same} ({len(rsi[keys0_list[0]])} terms)')
    else:
        print(f'  state {i}: NOT present')
