"""Measure on-disk pickle byte breakdown of a v5 ckpt.
What we want to know:
  Per beam state, how many bytes does each component take?
  - expr
  - resolved_subs
  - sub_accum
  - aux_flat (cu, ubm, rid, iraws)
"""
import pickle, sys, pickletools

ckpt = sys.argv[1]
with open(ckpt, 'rb') as f:
    d = pickle.load(f)

beam = d['beam']
print(f"step={d['step']}, n_beam={len(beam)}")

def sz(obj):
    return len(pickle.dumps(obj))

# Aggregate across all beam states
n = len(beam)
total_expr = sum(sz(s['expr']) for s in beam)
total_rs   = sum(sz(s['resolved_subs']) for s in beam)
total_sa   = sum(sz(s['sub_accum']) for s in beam)
total_aux  = sum(sz(s['aux_flat']) for s in beam)
total_other = sum(sz(s) for s in beam) - (total_expr+total_rs+total_sa+total_aux)

# Breakdown of aux_flat (cu, ubm, stable_rid, iraws, marker)
sum_cu = sum_ubm = sum_rid = sum_iraws = 0
for s in beam:
    a = s['aux_flat']
    if a is None: continue
    cu, ubm, rid, iraws = a[0], a[1], a[2], a[3]
    sum_cu    += sz(cu)
    sum_ubm   += sz(ubm)
    sum_rid   += sz(rid)
    sum_iraws += sz(iraws)

# Per-state averages
def fmt(b):
    if b >= 1024*1024: return f"{b/1024/1024:7.1f} MB"
    if b >= 1024:      return f"{b/1024:7.1f} KB"
    return f"{b:9d} B"

print(f"\nPer-beam-state component sizes (averaged across n={n}):")
print(f"  expr          {fmt(total_expr//n)}")
print(f"  resolved_subs {fmt(total_rs//n)}")
print(f"  sub_accum     {fmt(total_sa//n)}")
print(f"  aux_flat      {fmt(total_aux//n)}")
print(f"    aux.cu      {fmt(sum_cu//n)}")
print(f"    aux.ubm     {fmt(sum_ubm//n)}")
print(f"    aux.rid     {fmt(sum_rid//n)}")
print(f"    aux.iraws   {fmt(sum_iraws//n)}")
print(f"  other         {fmt(total_other//n)}")
print()
print(f"Totals across all {n} states:")
print(f"  expr          {fmt(total_expr)}")
print(f"  resolved_subs {fmt(total_rs)}")
print(f"  sub_accum     {fmt(total_sa)}")
print(f"  aux_flat      {fmt(total_aux)}")
print(f"  TOTAL         {fmt(total_expr+total_rs+total_sa+total_aux+total_other)}")

# RS stats
print(f"\nResolved_subs stats (best state):")
best_rs = beam[0]['resolved_subs']
n_keys = len(best_rs)
total_terms = sum(len(v) for v in best_rs.values())
print(f"  n_keys={n_keys}  rs_vsz(total terms)={total_terms}")
print(f"  avg value len = {total_terms/n_keys:.1f} terms" if n_keys else "  empty")

# Sample one RS value
if best_rs:
    sample_key = list(best_rs.keys())[0]
    sample_val = best_rs[sample_key]
    print(f"  sample value (key={sample_key[:4]}..., {len(sample_val)} terms):")
    for i, (k, v) in enumerate(list(sample_val.items())[:3]):
        print(f"    {k} -> {v}")
