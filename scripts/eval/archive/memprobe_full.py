"""Full memory-breakdown probe for v6 beam search.

Gated by env var SAILIR_MEMPROBE_FULL. When set, beam_search_v6.py invokes
`measure(env, beam, tabu_dict, step, out_path)` every
SAILIR_MEMPROBE_FULL_EVERY steps (default 10).

For each sample, writes ONE jsonl line with the deep-size breakdown of
every persistent structure:
  - env._raw_eq_cache (LRU cache; raw IBP equations)
  - per beam-slot: aux_flat.{cu, ubm, rid, iraws}, expr, resolved_subs,
    sub_accum, path
  - tabu_dict
Plus RSS from /proc, tracemalloc current/peak, and top-20 tracemalloc
allocators by size with traceback.

pympler.asizeof gives deep size walking referenced objects. Slow but
exact — the point here is accuracy, not speed.
"""
from __future__ import annotations

import json
import os
import time

import sys
import ctypes
import ctypes.util

from pympler.asizeof import asizeof

# Optional malloc_trim binding: forces glibc to return free-list pages to OS,
# giving a "real" persistent RSS. Without trim, glibc retains pages that were
# malloc'd and freed (e.g. transient torch forward activations).
try:
    _libc = ctypes.CDLL(ctypes.util.find_library('c'))
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except Exception:
    _malloc_trim = None


def malloc_trim_now():
    """Force glibc to give back free pages. Returns RSS in KB after trim."""
    if _malloc_trim is not None:
        try:
            _malloc_trim(0)
        except Exception:
            pass
    return _read_rss_kb()


def _classify_region(addr, sorted_ranges):
    """Find the smaps region containing this address. sorted_ranges is a
    list of (lo, hi, name_category) sorted by lo. Binary search.
    """
    import bisect
    # Build a key list of `lo` values for bisect.
    los = [r[0] for r in sorted_ranges]
    i = bisect.bisect_right(los, addr) - 1
    if i < 0:
        return 'unknown'
    lo, hi, name = sorted_ranges[i]
    if addr < hi:
        return name
    return 'unknown'


def _region_category(name):
    n = (name or '').strip()
    if n == '[heap]':
        return 'heap'
    if n == '':
        return 'anon'
    if n.endswith('.so') or '.so.' in n:
        return 'libs'
    if n.startswith('[stack'):
        return 'stack'
    if n.startswith('/'):
        return 'file'
    return 'other'


def _sample_attribution(items, address_ranges, n_sample=100):
    """For up to n_sample items, classify id(item) into smaps region cat.
    Returns Counter mapping category → fraction of samples landing there."""
    import random
    import collections
    if not items:
        return {}
    sample = random.sample(items, min(n_sample, len(items)))
    sorted_ranges = sorted(
        [(lo, hi, _region_category(name)) for lo, hi, name in address_ranges],
        key=lambda r: r[0])
    counts = collections.Counter()
    for obj in sample:
        counts[_classify_region(id(obj), sorted_ranges)] += 1
    n = sum(counts.values()) or 1
    return {k: round(v / n, 3) for k, v in counts.items()}


PYMALLOC_THRESHOLD = 512  # Python 3.10 SMALL_REQUEST_THRESHOLD


def _deep_size_dedup(obj, visited):
    """Hand-rolled iterative deep walk. Returns total bytes; dedupes by id()."""
    if obj is None:
        return 0
    sgz = sys.getsizeof
    stack = [obj]
    total = 0
    while stack:
        o = stack.pop()
        i = id(o)
        if i in visited:
            continue
        visited.add(i)
        try:
            total += sgz(o)
        except TypeError:
            continue
        t = type(o)
        if t is dict:
            for k, v in o.items():
                stack.append(k)
                stack.append(v)
        elif t is list or t is tuple or t is set or t is frozenset:
            for x in o:
                stack.append(x)
    return total


def _deep_size_split(obj, visited):
    """Deep walk that SPLITS bytes by allocator path:
      heap_bytes  = bytes from objects with sys.getsizeof > 512 (go via
                    glibc malloc → land in [heap] arena)
      arena_bytes = bytes from objects with sys.getsizeof <= 512 (go via
                    pymalloc → anon mmap'd arenas)

    For dict/list with separately-allocated data arrays, sys.getsizeof
    returns the FULL size including the data buffer (CPython's dict/list
    __sizeof__ accounts for the resize). So a 5 KB dict reports 5 KB and
    is correctly classified as heap-resident.

    Returns (total, heap_bytes, arena_bytes) where total = heap + arena.
    """
    if obj is None:
        return (0, 0, 0)
    sgz = sys.getsizeof
    stack = [obj]
    total = 0
    heap = 0
    arena = 0
    while stack:
        o = stack.pop()
        i = id(o)
        if i in visited:
            continue
        visited.add(i)
        try:
            sz = sgz(o)
        except TypeError:
            continue
        total += sz
        if sz > PYMALLOC_THRESHOLD:
            heap += sz
        else:
            arena += sz
        t = type(o)
        if t is dict:
            for k, v in o.items():
                stack.append(k)
                stack.append(v)
        elif t is list or t is tuple or t is set or t is frozenset:
            for x in o:
                stack.append(x)
    return (total, heap, arena)


def _read_rss_kb():
    """Read VmRSS from /proc/self/status. Linux-only."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _read_smaps_summary():
    """Walk /proc/self/smaps and return RSS by mmap-region category.

    Categories:
      file_backed_libs        — .so files (libtorch, libATen, libpython, etc.)
      file_backed_data        — other file-backed (model checkpoints, .py, etc.)
      heap_brk                — [heap] (glibc's sbrk-based main arena)
      stack                   — [stack] and [stack:*]
      anon_huge               — anonymous private regions > 64 MB
      anon_med                — anonymous private regions 1-64 MB
      anon_small              — anonymous private regions < 1 MB
      vsyscall_etc            — [vvar], [vdso], [vsyscall]
      other                   — anything else
    Plus `top_regions`: a list of the 10 largest individual regions
    (name, size_kb, rss_kb) for direct attribution of the unaccounted bytes.
    """
    cats = {
        'file_backed_libs_kb': 0,
        'file_backed_data_kb': 0,
        'heap_brk_kb': 0,
        'stack_kb': 0,
        'anon_huge_kb': 0,
        'anon_med_kb': 0,
        'anon_small_kb': 0,
        'vsyscall_etc_kb': 0,
        'other_kb': 0,
        'total_rss_kb': 0,
        'top_regions': [],
    }
    regions = []  # list of (name, size_kb, rss_kb)
    address_ranges = []  # list of (lo, hi, name) for object-id attribution
    cur_name = None
    cur_size = 0
    cur_rss = 0
    cur_lo = 0
    cur_hi = 0
    import re as _re
    HEADER_RE = _re.compile(r'^([0-9a-f]+)-([0-9a-f]+) [rwxsp\-]+ ')
    try:
        with open('/proc/self/smaps') as f:
            for line in f:
                m = HEADER_RE.match(line)
                if m:
                    if cur_name is not None:
                        regions.append((cur_name, cur_size, cur_rss))
                        address_ranges.append((cur_lo, cur_hi, cur_name))
                    cur_lo = int(m.group(1), 16)
                    cur_hi = int(m.group(2), 16)
                    parts = line.split(None, 5)
                    cur_name = parts[5].rstrip('\n') if len(parts) >= 6 else ''
                    cur_size = 0
                    cur_rss = 0
                elif line.startswith('Size:'):
                    cur_size = int(line.split()[1])
                elif line.startswith('Rss:'):
                    cur_rss = int(line.split()[1])
            if cur_name is not None:
                regions.append((cur_name, cur_size, cur_rss))
                address_ranges.append((cur_lo, cur_hi, cur_name))
    except OSError:
        return cats
    cats['_address_ranges'] = address_ranges
    for name, _sz, rss in regions:
        cats['total_rss_kb'] += rss
        n = (name or '').strip()
        if n.endswith('.so') or '.so.' in n:
            cats['file_backed_libs_kb'] += rss
        elif n == '[heap]':
            cats['heap_brk_kb'] += rss
        elif n.startswith('[stack'):
            cats['stack_kb'] += rss
        elif n in ('[vvar]', '[vdso]', '[vsyscall]'):
            cats['vsyscall_etc_kb'] += rss
        elif n == '':
            # Anonymous private region (Size in KB).
            if rss >= 64 * 1024:
                cats['anon_huge_kb'] += rss
            elif rss >= 1024:
                cats['anon_med_kb'] += rss
            else:
                cats['anon_small_kb'] += rss
        elif n.startswith('/'):
            cats['file_backed_data_kb'] += rss
        else:
            cats['other_kb'] += rss
    # Top-10 individual regions by RSS (truncate name for json size).
    regions.sort(key=lambda r: -r[2])
    cats['top_regions'] = [
        {'name': (r[0] or '<anon>')[-90:], 'size_kb': r[1], 'rss_kb': r[2]}
        for r in regions[:10]
    ]
    return cats


def _safe_asizeof(obj):
    if obj is None:
        return 0
    try:
        return int(asizeof(obj))
    except Exception:
        return -1


def _tracemalloc_top(n=20):
    """Return top N tracemalloc allocators by size (lineno-grouped).
    Each entry: dict(file, lineno, size_bytes, count).
    """
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            return []
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics('lineno')[:n]
        return [
            {
                'file': stat.traceback[0].filename,
                'lineno': stat.traceback[0].lineno,
                'size_bytes': int(stat.size),
                'count': int(stat.count),
            }
            for stat in stats
        ]
    except Exception:
        return []


def _tracemalloc_current_peak():
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            return (-1, -1)
        c, p = tracemalloc.get_traced_memory()
        return (int(c), int(p))
    except Exception:
        return (-1, -1)


def _tracemalloc_self_overhead():
    """Bytes used by tracemalloc's internal tracking structures.
    With 1M+ allocations × 10-frame tracebacks, this is non-trivial
    (typically 100-200 MB) and shows up as unaccounted C-side memory.
    """
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            return 0
        return int(tracemalloc.get_tracemalloc_memory())
    except Exception:
        return 0


_BASELINE = {
    'rss_at_import_kb': _read_rss_kb(),
    'rss_after_torch_load_kb': None,
    'rss_after_env_kb': None,
    'rss_after_model_kb': None,
    'model_param_bytes': None,
    'model_buffer_bytes': None,
}


def record_milestone(name):
    """Record an RSS milestone (called from main() at known points)."""
    if name in _BASELINE:
        _BASELINE[name] = _read_rss_kb()


def record_model(model):
    """Sum model param + buffer bytes once after model load."""
    try:
        p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        _BASELINE['model_param_bytes'] = int(p_bytes)
        _BASELINE['model_buffer_bytes'] = int(b_bytes)
    except Exception:
        pass


def baseline_snapshot():
    """Read once, write to header of jsonl on first measure()."""
    return dict(_BASELINE)


def measure(env, beam, tabu_dict, step, out_path, model=None,
            rss_phase_samples=None):
    """Take one full snapshot. Appends one jsonl record to out_path.

    If `model` is given, sums param + buffer bytes (torch tensors live
    outside the Python heap — tracemalloc doesn't see them, so we add
    them explicitly).

    `rss_phase_samples` is an optional dict of {phase_name: rss_kb}
    captured at phase boundaries within this step (start-of-step,
    before-model-fwd, peak-during-fwd, after-model-fwd, end-of-step).
    Captures transient peaks so we can compare against the
    end-of-step quiescent RSS used by the persistent-bytes sum.
    """
    t0 = time.time()
    # Minimal-cost path: skip pympler.asizeof and tracemalloc snapshots
    # (both grow O(traced objects) and dominate t_meas on big runs).
    # smaps + RSS + trim + cheap counters is enough to pin RSS attribution.
    # ORDER MATTERS: read both rss_kb and smaps BEFORE malloc_trim, so
    # the categorised breakdown matches VmRSS (otherwise trim drops the
    # heap_brk region between the two reads and the sums disagree by the
    # trim amount).
    import gc as _gc
    _gc.collect()
    rss_kb = _read_rss_kb()
    smaps = _read_smaps_summary()                  # cheap: one /proc read
    rss_after_trim_kb = malloc_trim_now()
    glibc_retained_kb = max(0, rss_kb - rss_after_trim_kb)
    tm_cur, tm_peak = _tracemalloc_current_peak()  # cheap: cached counter
    tm_self_bytes = _tracemalloc_self_overhead()   # cheap: cached counter

    # gc.get_objects walk is O(all_objects) and dominates t_meas at later
    # steps. Skipped — smaps gives the C-side RSS attribution we need.
    torch_live_tensor_bytes = 0
    torch_live_tensor_count = 0
    np_live_bytes = 0
    np_live_count = 0

    # env._raw_eq_cache: LRU or plain dict
    cache = getattr(env, '_raw_eq_cache', None)
    if cache is None:
        cache_count = 0
        cache_bytes = 0
        cache_stats = None
    else:
        cache_count = len(cache)
        cache_bytes = 0  # deep size below, in env_cache_bytes_deep
        cache_stats = getattr(cache, '_stats', None)

    # Per-structure deep size, each with a FRESH visited set. This gives the
    # TRUE marginal size of each structure (what would be freed if we
    # released it). Cross-structure sharing means the SUM may double-count
    # (e.g. shared tuple keys count under both cache and cu), but each
    # individual row and its Δ tell the unambiguous truth about whether
    # that specific structure is growing.
    env_cache_obj = getattr(env, '_raw_eq_cache', None)
    # If the cache is the LRU wrapper class, its data lives in
    # self._lru (OrderedDict) + self._meta (dict). Otherwise it's a plain
    # dict that the walker handles directly.
    if hasattr(env_cache_obj, '_lru'):
        v = set()
        t_lru, h_lru, a_lru = _deep_size_split(env_cache_obj._lru, v)
        t_meta, h_meta, a_meta = _deep_size_split(env_cache_obj._meta, v)
        env_cache_total = t_lru + t_meta + sys.getsizeof(env_cache_obj)
        env_cache_heap = h_lru + h_meta
        env_cache_arena = a_lru + a_meta + sys.getsizeof(env_cache_obj)
    else:
        env_cache_total, env_cache_heap, env_cache_arena = _deep_size_split(env_cache_obj, set())
    env_cache_bytes_deep = env_cache_total
    tabu_total, tabu_heap, tabu_arena = _deep_size_split(tabu_dict, set())
    tabu_bytes_deep = tabu_total

    # Per-structure heap/arena split — proves which structure contributes
    # to which smaps category.
    split = {
        'cache_heap': env_cache_heap, 'cache_arena': env_cache_arena,
        'tabu_heap': tabu_heap, 'tabu_arena': tabu_arena,
        'cu_heap': 0, 'cu_arena': 0,
        'iraws_heap': 0, 'iraws_arena': 0,
        'rid_heap': 0, 'rid_arena': 0,
    }

    # Sample N random items from each structure and check id()→smaps region.
    # Gives airtight causal attribution: if 100% of cache value-dict addresses
    # fall in [heap], then growing cache literally is growing [heap].
    addr_ranges = smaps.get('_address_ranges') or []
    sample_attribution = {}
    try:
        cache_items = (list(env_cache_obj._lru.values())
                        if hasattr(env_cache_obj, '_lru')
                        else list(env_cache_obj.values()) if env_cache_obj
                        else [])
        sample_attribution['cache_values'] = _sample_attribution(
            cache_items, addr_ranges, n_sample=200)
    except Exception:
        pass
    try:
        # Sample cu dicts from beam slots
        all_cu_dicts = []
        for s in beam:
            af = getattr(s, 'aux_flat', None)
            if af is not None:
                all_cu_dicts.extend(af[0])
        sample_attribution['cu_dicts'] = _sample_attribution(
            all_cu_dicts, addr_ranges, n_sample=200)
    except Exception:
        pass
    try:
        # Sample tabu set objects
        tabu_sets = list(tabu_dict.values()) if tabu_dict else []
        sample_attribution['tabu_sets'] = _sample_attribution(
            tabu_sets, addr_ranges, n_sample=200)
    except Exception:
        pass

    # Strip the heavyweight _address_ranges from smaps before jsonl write.
    smaps.pop('_address_ranges', None)

    per_field = {
        'aux_cu': 0,
        'aux_ubm': 0,
        'aux_rid': 0,
        'aux_iraws': 0,
        'expr': 0,
        'resolved_subs': 0,
        'sub_accum': 0,
        'path': 0,
    }
    per_field_counts = {
        'aux_cu_entries': 0,
        'aux_ubm_entries': 0,
        'aux_rid_entries': 0,
        'aux_iraws_entries': 0,
        'expr_keys_total': 0,
        'resolved_subs_keys_total': 0,
        'sub_accum_keys_total': 0,
        'path_len_total': 0,
    }
    n_beam = len(beam)
    # For each PER-FIELD bucket, use a SEPARATE visited set per bucket so
    # we sum sizes across beam slots correctly (within bucket: dedupe
    # shared sub-objects across beams; across buckets: count independently).
    v_cu = set(); v_ubm = set(); v_rid = set(); v_iraws = set()
    v_expr = set(); v_rs = set(); v_sa = set(); v_path = set()
    for s in beam:
        af = getattr(s, 'aux_flat', None)
        if af is not None:
            cu, ubm, rid, iraws = af
            t, h, a = _deep_size_split(cu, v_cu)
            per_field['aux_cu'] += t
            split['cu_heap'] += h
            split['cu_arena'] += a
            per_field['aux_ubm'] += _deep_size_dedup(ubm, v_ubm)
            t, h, a = _deep_size_split(rid, v_rid)
            per_field['aux_rid'] += t
            split['rid_heap'] += h
            split['rid_arena'] += a
            t, h, a = _deep_size_split(iraws, v_iraws)
            per_field['aux_iraws'] += t
            split['iraws_heap'] += h
            split['iraws_arena'] += a
            per_field_counts['aux_cu_entries'] += len(cu)
            per_field_counts['aux_ubm_entries'] += len(ubm)
            per_field_counts['aux_rid_entries'] += len(rid)
            per_field_counts['aux_iraws_entries'] += len(iraws)
        e = getattr(s, 'expr', None)
        if e is not None:
            per_field['expr'] += _deep_size_dedup(e, v_expr)
            per_field_counts['expr_keys_total'] += len(e)
        rs = getattr(s, 'resolved_subs', None)
        if rs is not None:
            per_field['resolved_subs'] += _deep_size_dedup(rs, v_rs)
            per_field_counts['resolved_subs_keys_total'] += len(rs)
        sa = getattr(s, 'sub_accum', None)
        if sa is not None:
            per_field['sub_accum'] += _deep_size_dedup(sa, v_sa)
            per_field_counts['sub_accum_keys_total'] += len(sa)
        p = getattr(s, 'path', None)
        if p is not None:
            per_field['path'] += _deep_size_dedup(p, v_path)
            per_field_counts['path_len_total'] += len(p)

    tabu_bytes = tabu_bytes_deep
    tabu_buckets = len(tabu_dict) if tabu_dict else 0
    tabu_entries = sum(len(v) for v in tabu_dict.values()) if tabu_dict else 0

    # Torch model param + buffer bytes (C-side, NOT seen by tracemalloc).
    model_param_bytes = 0
    model_buffer_bytes = 0
    if model is not None:
        try:
            model_param_bytes = int(sum(p.numel() * p.element_size()
                                         for p in model.parameters()))
            model_buffer_bytes = int(sum(b.numel() * b.element_size()
                                          for b in model.buffers()))
        except Exception:
            pass

    sum_persistent_bytes = (
        cache_bytes
        + sum(per_field.values())
        + tabu_bytes
        + model_param_bytes
        + model_buffer_bytes
    )

    record = {
        'step': step,
        't_measure_s': 0.0,  # filled below
        'rss_mb': (rss_kb // 1024) if rss_kb > 0 else -1,
        'rss_kb': rss_kb,
        'rss_after_trim_kb': rss_after_trim_kb,
        'glibc_retained_kb': glibc_retained_kb,
        'torch_live_tensor_bytes': torch_live_tensor_bytes,
        'torch_live_tensor_count': torch_live_tensor_count,
        'np_live_bytes': np_live_bytes,
        'np_live_count': np_live_count,
        'tracemalloc_current_bytes': tm_cur,
        'tracemalloc_peak_bytes': tm_peak,
        'tracemalloc_self_bytes': tm_self_bytes,
        'smaps': smaps,
        'sample_attribution': sample_attribution,
        'allocator_split': split,
        'env_raw_eq_cache_count': cache_count,
        'env_raw_eq_cache_bytes': env_cache_bytes_deep,
        'env_raw_eq_cache_stats': cache_stats,
        'per_field_bytes': per_field,
        'per_field_counts': per_field_counts,
        'model_param_bytes': model_param_bytes,
        'model_buffer_bytes': model_buffer_bytes,
        'sum_per_beam_bytes': sum(per_field.values()),
        'sum_persistent_bytes': sum_persistent_bytes,
        'tabu_dict_bytes': tabu_bytes,
        'tabu_dict_buckets': tabu_buckets,
        'tabu_dict_entries': tabu_entries,
        'n_beam': n_beam,
        'rss_phase_samples': rss_phase_samples,
        'baseline_milestones': baseline_snapshot(),
        'unaccounted_mb': max(
            0,
            ((rss_kb * 1024) - sum_persistent_bytes) // (1024 * 1024)
        ) if rss_kb > 0 else -1,
        'tracemalloc_top20': _tracemalloc_top(20),
    }
    record['t_measure_s'] = round(time.time() - t0, 3)

    with open(out_path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')

    return record
