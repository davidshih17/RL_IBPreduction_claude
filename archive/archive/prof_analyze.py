"""Summarize a cProfile dump: top functions by SELF time (tottime), and bucket
total self-time into MODEL (torch/neural net) vs PACKED-KERNELS (the math loops
we'd Cythonize) vs OTHER. Tells us the Cython payoff ceiling + the targets.
"""
import pstats
import sys

PACKED = (  # the packed math kernels + their callees (Cython candidates)
    'substitute_one', 'apply_resolved_subs', 'union_bitmask', '_resolve_packed',
    '_get_packed_sub', 'compute_indirect_substituted_incremental_packed',
    'enumerate_valid_actions_with_indirect_cache_packed',
    'add_sub_to_resolved_packed', 'strip_passenger_packed',
    'apply_resolved_subs_dict_x_packed', 'to_dict', 'from_dict', 'get_tuple',
    'get_id', '_aux_to_result', '_v7_aux_pack', '_v7_as_dict',
    'compute_indirect_substituted_with_aux', 'apply_resolved_subs_batch',
    'cached_union_bitmask', 'get_raw_equation', 'solve_ibp_for',
    'apply_substitution_v5', 'get_non_masters', 'weight', 'searchsorted',
    'concatenate', 'fromiter',
)
MODEL = ('forward', 'torch', 'conv', 'linear', 'matmul', 'prepare_batched_input',
         'embedding', 'softmax', 'relu', 'addmm', 'tensor', 'aten')


def bucket(name):
    low = name.lower()
    if any(m in low for m in MODEL):
        return 'MODEL'
    if any(p in name for p in PACKED):
        return 'PACKED'
    return 'OTHER'


def main():
    st = pstats.Stats(sys.argv[1])
    st.sort_stats('tottime')
    total_self = sum(v[2] for v in st.stats.values())  # tottime
    print(f"\n=== cProfile summary (total self-time {total_self:.1f}s) ===\n")
    sums = {'MODEL': 0.0, 'PACKED': 0.0, 'OTHER': 0.0}
    rows = []
    for (fn, line, name), v in st.stats.items():
        cc, nc, tt, ct, callers = v
        b = bucket(name)
        sums[b] += tt
        rows.append((tt, ct, nc, b, name, fn.split('/')[-1], line))
    print("bucket totals (self-time):")
    for b in ('MODEL', 'PACKED', 'OTHER'):
        print(f"  {b:7s} {sums[b]:7.1f}s  ({100*sums[b]/max(total_self,1e-9):4.1f}%)")
    print("\ntop 30 functions by SELF time:")
    print(f"  {'self_s':>8} {'cum_s':>8} {'ncalls':>9}  {'bkt':5} name (file:line)")
    for tt, ct, nc, b, name, f, line in sorted(rows, reverse=True)[:30]:
        print(f"  {tt:8.2f} {ct:8.2f} {nc:9d}  {b:5} {name} ({f}:{line})")


if __name__ == '__main__':
    main()
