"""Analyze memprobe_full jsonl with non-double-counting attribution.

RSS = baseline(at import) + python_heap_growth + torch_tensor_bytes
       + np_array_bytes + glibc_retained_kb + unaccounted

  baseline      = rss_at_import_kb               (libs + initial Python heap)
  python_heap   = tracemalloc_current_bytes      (Python objects, deduped)
  torch_tensors = sum(t.numel*t.element_size for t in gc + tensors)
  np_arrays     = sum(a.nbytes for a in np arrays)
  glibc_retained= rss - rss_after_trim          (transient free pool)

Plus a secondary "where is python heap going" table from per_field_bytes
(which overcounts due to shared int references in tuple keys — for relative
attribution only, not absolute).
"""
import json
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: analyze_memprobe.py <memprobe.jsonl>")
        sys.exit(1)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print("(no rows)")
        return
    MB = 1024 * 1024
    KB_PER_MB = 1024

    baseline_kb = rows[0]['baseline_milestones'].get('rss_at_import_kb', 0)
    baseline_mb = baseline_kb // KB_PER_MB

    print(f"== rss_at_import baseline = {baseline_mb} MB (libs + initial python heap) ==")
    print()
    # smaps-based attribution: file_libs + heap + stack + anon_huge + anon_med
    # + anon_small + vsyscall + other == total smaps rss == kernel RSS.
    print(f"{'step':>5} {'rss':>5} | "
          f"{'libs':>5} {'heap':>5} {'stack':>5} {'anonH':>5} {'anonM':>5} "
          f"{'anonS':>5} {'other':>5} | "
          f"{'smaps_sum':>9} {'%':>4}  {'t_m':>5}")
    for r in rows:
        rss = r.get('rss_mb', -1)
        trim_kb = r.get('rss_after_trim_kb', 0)
        glibc_kb = r.get('glibc_retained_kb', 0)
        trim_mb = trim_kb // KB_PER_MB if trim_kb > 0 else -1
        glibc_mb = glibc_kb // KB_PER_MB if glibc_kb > 0 else 0
        pyheap_mb = r['tracemalloc_current_bytes'] // MB if r['tracemalloc_current_bytes'] > 0 else 0
        tm_oh_mb = r.get('tracemalloc_self_bytes', 0) // MB
        torch_mb = r.get('torch_live_tensor_bytes', 0) // MB
        np_mb = r.get('np_live_bytes', 0) // MB
        cache_n = r['env_raw_eq_cache_count']
        cu_est_mb = r['per_field_bytes']['aux_cu'] // MB  # overcounts; show for relative attribution
        iraws_e = r['per_field_counts'].get('aux_iraws_entries', 0)
        t_m = r.get('t_measure_s', 0)

        sm = r.get('smaps') or {}
        libs_mb = sm.get('file_backed_libs_kb', 0) // KB_PER_MB
        heap_mb = sm.get('heap_brk_kb', 0) // KB_PER_MB
        stack_mb = sm.get('stack_kb', 0) // KB_PER_MB
        anonH_mb = sm.get('anon_huge_kb', 0) // KB_PER_MB
        anonM_mb = sm.get('anon_med_kb', 0) // KB_PER_MB
        anonS_mb = sm.get('anon_small_kb', 0) // KB_PER_MB
        other_mb = (sm.get('file_backed_data_kb', 0) + sm.get('vsyscall_etc_kb', 0)
                    + sm.get('other_kb', 0)) // KB_PER_MB
        smaps_sum = libs_mb + heap_mb + stack_mb + anonH_mb + anonM_mb + anonS_mb + other_mb
        pct = (smaps_sum * 100 // rss) if rss > 0 else -1
        print(f"{r['step']:>5} {rss:>5} | "
              f"{libs_mb:>5} {heap_mb:>5} {stack_mb:>5} {anonH_mb:>5} {anonM_mb:>5} "
              f"{anonS_mb:>5} {other_mb:>5} | "
              f"{smaps_sum:>9} {pct:>4}  {t_m:>5.1f}")
    # Growth attribution: Δ between consecutive samples for each smaps
    # category. Sum of category Δ == RSS Δ by construction.
    if len(rows) >= 2:
        print()
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            sa = a.get('smaps') or {}
            sb = b.get('smaps') or {}
            d_rss = b['rss_mb'] - a['rss_mb']
            d_libs = (sb.get('file_backed_libs_kb', 0) - sa.get('file_backed_libs_kb', 0)) // KB_PER_MB
            d_heap = (sb.get('heap_brk_kb', 0) - sa.get('heap_brk_kb', 0)) // KB_PER_MB
            d_anH = (sb.get('anon_huge_kb', 0) - sa.get('anon_huge_kb', 0)) // KB_PER_MB
            d_anM = (sb.get('anon_med_kb', 0) - sa.get('anon_med_kb', 0)) // KB_PER_MB
            d_anS = (sb.get('anon_small_kb', 0) - sa.get('anon_small_kb', 0)) // KB_PER_MB
            d_other = ((sb.get('file_backed_data_kb', 0) + sb.get('vsyscall_etc_kb', 0)
                         + sb.get('other_kb', 0))
                        - (sa.get('file_backed_data_kb', 0) + sa.get('vsyscall_etc_kb', 0)
                           + sa.get('other_kb', 0))) // KB_PER_MB
            d_smaps = d_libs + d_heap + d_anH + d_anM + d_anS + d_other
            print(f"=== GROWTH step {a['step']} → {b['step']} ===")
            print(f"  RSS:        +{d_rss:>5} MB")
            print(f"  libs:       +{d_libs:>5} MB")
            print(f"  glibc heap: +{d_heap:>5} MB")
            print(f"  anon_huge:  +{d_anH:>5} MB")
            print(f"  anon_med:   +{d_anM:>5} MB")
            print(f"  anon_small: +{d_anS:>5} MB")
            print(f"  other:      +{d_other:>5} MB")
            print(f"  --- smaps sum Δ: +{d_smaps:>5} MB  (RSS Δ: +{d_rss} MB; "
                  f"diff = {d_rss - d_smaps} MB)")
    print()
    print("=== Python-heap structure attribution (deep-size, within-bucket dedup) ===")
    print(f"{'step':>5}  {'cache_MB':>8} {'cu_MB':>6} {'iraws_MB':>8} {'rid_MB':>6} {'tabu_MB':>7}  "
          f"{'cache_n':>8} {'cu_n':>7} {'iraws_n':>8} {'tabu_n':>7}")
    for r in rows:
        pf = r['per_field_bytes']
        pfc = r.get('per_field_counts') or {}
        cache_b = r.get('env_raw_eq_cache_bytes', 0) // MB
        cu = pf['aux_cu'] // MB
        iraws = pf['aux_iraws'] // MB
        rid = pf['aux_rid'] // MB
        tabu_mb = r.get('tabu_dict_bytes', 0) // MB
        cache_n = r.get('env_raw_eq_cache_count', 0)
        cu_n = pfc.get('aux_cu_entries', 0)
        iraws_n = pfc.get('aux_iraws_entries', 0)
        tabu_n = r.get('tabu_dict_entries', 0)
        print(f"{r['step']:>5}  {cache_b:>8} {cu:>6} {iraws:>8} {rid:>6} {tabu_mb:>7}  "
              f"{cache_n:>8} {cu_n:>7} {iraws_n:>8} {tabu_n:>7}")
    print()
    print("=== Δ-by-structure ===")
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        pfa, pfb = a['per_field_bytes'], b['per_field_bytes']
        def Δ(d):
            return d // MB
        d_cache = Δ(b.get('env_raw_eq_cache_bytes', 0) - a.get('env_raw_eq_cache_bytes', 0))
        d_cu = Δ(pfb['aux_cu'] - pfa['aux_cu'])
        d_iraws = Δ(pfb['aux_iraws'] - pfa['aux_iraws'])
        d_rid = Δ(pfb['aux_rid'] - pfa['aux_rid'])
        d_ubm = Δ(pfb['aux_ubm'] - pfa['aux_ubm'])
        d_rs = Δ(pfb['resolved_subs'] - pfa['resolved_subs'])
        d_expr = Δ(pfb['expr'] - pfa['expr'])
        d_sa = Δ(pfb['sub_accum'] - pfa['sub_accum'])
        d_path = Δ(pfb['path'] - pfa['path'])
        d_tabu = Δ(b.get('tabu_dict_bytes', 0) - a.get('tabu_dict_bytes', 0))
        d_sum = d_cache + d_cu + d_iraws + d_rid + d_ubm + d_rs + d_expr + d_sa + d_path + d_tabu
        print(f"step {a['step']} → {b['step']}:  cache +{d_cache:>4} | cu +{d_cu:>4} | iraws +{d_iraws:>4} | "
              f"rid +{d_rid:>3} | ubm +{d_ubm:>3} | rs +{d_rs:>3} | expr +{d_expr:>3} | "
              f"sa +{d_sa:>3} | path +{d_path:>3} | tabu +{d_tabu:>3}  → +{d_sum} MB")

    print()
    print("=== heap vs arena split per structure (sgz>512 → [heap]) ===")
    print(f"{'step':>5}  {'cache_h':>7} {'cache_a':>7}  {'cu_h':>6} {'cu_a':>6}  "
          f"{'iraws_h':>7} {'iraws_a':>7}  {'rid_h':>5} {'rid_a':>5}  "
          f"{'tabu_h':>6} {'tabu_a':>6}  {'sum_h':>5} {'sum_a':>5}")
    for r in rows:
        sp = r.get('allocator_split') or {}
        ch = sp.get('cache_heap', 0) // MB
        ca = sp.get('cache_arena', 0) // MB
        cuh = sp.get('cu_heap', 0) // MB
        cua = sp.get('cu_arena', 0) // MB
        irh = sp.get('iraws_heap', 0) // MB
        ira = sp.get('iraws_arena', 0) // MB
        rdh = sp.get('rid_heap', 0) // MB
        rda = sp.get('rid_arena', 0) // MB
        tbh = sp.get('tabu_heap', 0) // MB
        tba = sp.get('tabu_arena', 0) // MB
        sh = ch + cuh + irh + rdh + tbh
        sa = ca + cua + ira + rda + tba
        print(f"{r['step']:>5}  {ch:>7} {ca:>7}  {cuh:>6} {cua:>6}  "
              f"{irh:>7} {ira:>7}  {rdh:>5} {rda:>5}  "
              f"{tbh:>6} {tba:>6}  {sh:>5} {sa:>5}")
    print()
    print("=== sample id() → smaps region (header-address attribution) ===")
    for r in rows:
        sa = r.get('sample_attribution') or {}
        line = f"step {r['step']}: "
        for struct, fracs in sa.items():
            frac_str = ' '.join(f"{cat}={v}" for cat, v in fracs.items())
            line += f"\n  {struct}: {frac_str}"
        print(line)
    print()
    print("=== top regions (last sample) ===")
    last = rows[-1]
    for r in (last.get('smaps') or {}).get('top_regions', []):
        print(f"  {r['rss_kb']:>10} kb  {r['name']}")


if __name__ == '__main__':
    main()
