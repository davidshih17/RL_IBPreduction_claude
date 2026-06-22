"""Quick outcome breakdown for workers that finished in the last N hours.

We pick workers whose .log mtime falls in [now-Nh, now-5min] and bucket by:
  - SUCCESS                   (result.pkl, success=True)
  - TABU_TRAP                 (result.pkl, success=False, best_nm>=1)
  - SUCCESS_FALSE_OTHER       (result.pkl, success=False, best_nm<1)
  - ORCH_KILL_ZOMBIE          (no result.pkl, .log has 'via condor_rm')
  - OOM_SILENT_SIGKILL        (no result.pkl, .log has 'signal 9')
  - OOM_PYTORCH               (no result.pkl, .err matches alloc patterns)
  - OOM_OTHER                 (no result.pkl, .err has MemoryError etc.)
  - TRACEBACK                 (no result.pkl, .err has Traceback)
  - NO_RESULT_NO_DIAG         (no result.pkl, no recognized diagnostic)

Also separates MANUAL_OOM from ORCH_KILL_ZOMBIE using manual_oom_killed.csv.

Usage:
    python recent_outcomes.py <sweep_root> --hours 8
"""
import argparse, os, pickle, re, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('--hours', type=float, default=8.0)
    p.add_argument('--min-age-sec', type=float, default=300,
                   help='Skip files younger than this — they may still be writing.')
    args = p.parse_args()

    log_dir = os.path.join(args.sweep_root, 'work', 'logs')
    res_dir = os.path.join(args.sweep_root, 'work', 'results')
    now = time.time()
    win_start = now - args.hours * 3600
    win_end = now - args.min_age_sec

    # Read manual-OOM kill list
    manual_oom = set()
    mc = os.path.join(args.sweep_root, 'manual_oom_killed.csv')
    if os.path.exists(mc):
        with open(mc) as fp:
            next(fp, None)
            for line in fp:
                cols = line.rstrip().split(',')
                if len(cols) >= 6:
                    manual_oom.add(cols[5])

    counts = {}
    def bump(cat):
        counts[cat] = counts.get(cat, 0) + 1

    t0 = time.time()
    n = 0
    for fn in os.listdir(log_dir):
        if not fn.endswith('.log'):
            continue
        log_path = os.path.join(log_dir, fn)
        try:
            mt = os.path.getmtime(log_path)
        except OSError:
            continue
        if not (win_start <= mt <= win_end):
            continue
        base = fn[:-4]
        n += 1
        pkl_path = os.path.join(res_dir, base + '.pkl')
        if os.path.exists(pkl_path):
            try:
                r = pickle.load(open(pkl_path, 'rb'))
            except Exception:
                bump('PKL_UNREADABLE')
                continue
            if r.get('success'):
                bump('SUCCESS')
                continue
            best_nm = r.get('best_n_non_masters')
            if best_nm is not None and best_nm >= 1:
                bump('TABU_TRAP')
            else:
                bump('SUCCESS_FALSE_OTHER')
            continue
        # No result.pkl — read .log to distinguish kill vs OOM
        log_text = ''
        try:
            log_text = open(log_path).read(32 * 1024)
        except Exception:
            pass
        if 'via condor_rm' in log_text:
            if base in manual_oom:
                bump('MANUAL_OOM')
            else:
                bump('ORCH_KILL_ZOMBIE')
            continue
        if 'Abnormal termination (signal 9)' in log_text:
            bump('OOM_SILENT_SIGKILL')
            continue
        err_path = os.path.join(log_dir, base + '.err')
        try:
            err_text = open(err_path).read(64 * 1024)
        except Exception:
            err_text = ''
        if 'Cannot allocate memory' in err_text or 'alloc_cpu.cpp' in err_text:
            bump('OOM_PYTORCH')
        elif any(k in err_text for k in ('MemoryError', 'std::bad_alloc',
                                          'OutOfMemory')):
            bump('OOM_OTHER')
        elif 'Traceback (most recent call last)' in err_text:
            bump('TRACEBACK')
        else:
            bump('NO_RESULT_NO_DIAG')

    elapsed = time.time() - t0
    print(f'Window: last {args.hours} h (excluding youngest {args.min_age_sec}s)')
    print(f'Workers in window: {n}    scan time: {elapsed:.1f}s')
    print()
    print(f'{"category":<22s} {"count":>6s}')
    for k in sorted(counts, key=lambda c: -counts[c]):
        print(f'  {k:<22s} {counts[k]:>5d}')


if __name__ == '__main__':
    main()
