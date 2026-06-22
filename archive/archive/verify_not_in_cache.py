"""Take the 'NOT in cache' non-masters identified by replay and classify
each by direct file evidence:

  - SUBMITTED + DIED      : has .log AND .err but no .pkl. Read .log to see
                            whether condor_rm or signal 9 or normal-exit
                            (PyTorch OOM raises RuntimeError → exit code 1
                            → "Normal termination (return value 1)" in .log).
  - NEVER_SUBMITTED       : no .log file (integral never had a worker)
  - SUBMITTED + RUNNING   : has .log without 'Job terminated' or 'aborted'
                            event (currently in flight).

Walks every .log file once, indexes by integral, then iterates the input
non-master list.

Usage:
    python verify_not_in_cache.py <sweep_root> <start_integral> [--limit N]
"""
import argparse, glob, os, pickle, re, sys, time
from collections import Counter


def parse_integral_from_basename(base):
    parts = base.split('_')
    nums = []
    for p in reversed(parts):
        try:
            int(p)
        except ValueError:
            break
        nums.append(p)
        if len(nums) >= 11:
            break
    if len(nums) < 11:
        return None
    nums = nums[:11][::-1]
    return tuple(int(x) for x in nums)


def apply_substitutions_with_skip(expr, cache, prime, max_iter=200000):
    """Issue-1-fixed: skip identity entries to avoid the infinite loop."""
    changed = True
    iters = 0
    while changed:
        changed = False
        iters += 1
        new_expr = {}
        for integ, coeff in expr.items():
            if coeff == 0:
                continue
            if integ in cache:
                v = cache[integ]
                if v == {integ: 1}:
                    new_expr[integ] = (new_expr.get(integ, 0) + coeff) % prime
                    continue
                for k, c in v.items():
                    if c == 0:
                        continue
                    new_expr[k] = (new_expr.get(k, 0) + coeff * c) % prime
                changed = True
            else:
                new_expr[integ] = (new_expr.get(integ, 0) + coeff) % prime
        expr = {k: v for k, v in new_expr.items() if v != 0}
        if iters > max_iter:
            print(f'WARN apply_substitutions hit {max_iter} iters')
            break
    return expr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('start_integral')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    work_dir = os.path.join(args.sweep_root, 'work')
    res_dir = os.path.join(work_dir, 'results')
    log_dir = os.path.join(work_dir, 'logs')
    start_int = tuple(int(x) for x in args.start_integral.split(','))

    t0 = time.time()

    # ── Step 1: rebuild cache from disk ──────────────────────────────────
    print(f'[{time.time()-t0:5.1f}s] reading result.pkl files ...', flush=True)
    cache = {}
    n_pkl = 0
    for f in glob.glob(os.path.join(res_dir, '*.pkl')):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        ig = r.get('original_integral')
        if ig is None:
            continue
        if r.get('success'):
            cache[ig] = r.get('final_expr', {})
        else:
            cache[ig] = {ig: 1}
        n_pkl += 1
    print(f'[{time.time()-t0:5.1f}s]   cache: {n_pkl} entries', flush=True)

    # ── Step 2: replay to get active expr ────────────────────────────────
    print(f'[{time.time()-t0:5.1f}s] replaying substitutions ...', flush=True)
    expr = apply_substitutions_with_skip({start_int: 1}, cache, args.prime)
    not_in_cache = [ig for ig in expr if ig not in cache]
    print(f'[{time.time()-t0:5.1f}s]   active expr: {len(expr)}, '
          f'non-masters not in cache: {len(not_in_cache)}', flush=True)

    # ── Step 3: index all .log files by integral ─────────────────────────
    print(f'[{time.time()-t0:5.1f}s] indexing .log files ...', flush=True)
    log_by_integ = {}  # integ -> list of basenames with that integ
    for fn in os.listdir(log_dir):
        if not fn.endswith('.log'):
            continue
        base = fn[:-4]
        ig = parse_integral_from_basename(base)
        if ig is None:
            continue
        log_by_integ.setdefault(ig, []).append(base)
    print(f'[{time.time()-t0:5.1f}s]   {len(log_by_integ)} integrals have at '
          f'least one .log', flush=True)

    # ── Step 4: classify each "not in cache" non-master ──────────────────
    print(f'[{time.time()-t0:5.1f}s] classifying {len(not_in_cache)} '
          f'no-cache integrals ...', flush=True)
    by_cat = Counter()
    samples = {}
    for ig in not_in_cache:
        if ig not in log_by_integ:
            by_cat['NEVER_SUBMITTED'] += 1
            samples.setdefault('NEVER_SUBMITTED', []).append(ig)
            continue
        # Look at the MOST RECENT .log for this integ
        bases = log_by_integ[ig]
        latest_base = max(bases, key=lambda b: os.path.getmtime(
            os.path.join(log_dir, b + '.log')) if os.path.exists(
                os.path.join(log_dir, b + '.log')) else 0)
        log_path = os.path.join(log_dir, latest_base + '.log')
        try:
            log_text = open(log_path).read(32 * 1024)
        except Exception:
            by_cat['LOG_UNREADABLE'] += 1
            continue

        # Has the job terminated at all? Look for the standard events.
        # 005 = "Job terminated", 009 = "Job was aborted",
        # 004 = "Job was evicted"
        if '009 (' in log_text or 'Job was aborted' in log_text:
            if 'via condor_rm' in log_text:
                by_cat['CONDOR_RM_ABORT'] += 1
                samples.setdefault('CONDOR_RM_ABORT', []).append(ig)
            else:
                by_cat['OTHER_ABORT'] += 1
                samples.setdefault('OTHER_ABORT', []).append(ig)
        elif 'Abnormal termination (signal 9)' in log_text:
            by_cat['SIGKILL_NO_PKL'] += 1
            samples.setdefault('SIGKILL_NO_PKL', []).append(ig)
        elif 'Job terminated' in log_text or '005 (' in log_text:
            # Job terminated cleanly but no pkl — read .err to see why
            err_path = os.path.join(log_dir, latest_base + '.err')
            err_text = ''
            try:
                err_text = open(err_path).read(64 * 1024)
            except Exception:
                pass
            if ('Cannot allocate memory' in err_text
                    or 'alloc_cpu.cpp' in err_text):
                by_cat['OOM_PYTORCH'] += 1
            elif any(k in err_text for k in ('MemoryError', 'std::bad_alloc',
                                              'OutOfMemory')):
                by_cat['OOM_OTHER'] += 1
            elif 'Traceback (most recent call last)' in err_text:
                by_cat['TRACEBACK_NO_PKL'] += 1
            else:
                by_cat['TERMINATED_NO_DIAG'] += 1
                samples.setdefault('TERMINATED_NO_DIAG', []).append(ig)
        else:
            # No termination event — worker may still be running
            by_cat['STILL_RUNNING_OR_INDETERMINATE'] += 1
            samples.setdefault('STILL_RUNNING_OR_INDETERMINATE', []).append(ig)

    print()
    print(f'=== classification of {len(not_in_cache)} not-in-cache non-masters ===')
    for k in sorted(by_cat, key=lambda c: -by_cat[c]):
        print(f'  {k:<32s} {by_cat[k]:>5d}')
    print()
    for cat, igs in samples.items():
        if len(igs) <= 3:
            for ig in igs[:3]:
                print(f'  {cat} sample: I{list(ig)}')


if __name__ == '__main__':
    main()
