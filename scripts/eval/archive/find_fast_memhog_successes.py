"""Walk results/*.pkl, keep workers with success=True AND
peak_memory_kb > <threshold MB>, sorted by elapsed time ascending.

Used to pick fast bit-identicality test integrals — ones that DID reduce
but pushed peak memory > 10 GB, so they exercise the memory-bloat code
path AND finish quickly enough to make a tight test loop.

Usage:
    python find_fast_memhog_successes.py <work_dir> [--min-mb 10000] [--top 30]
"""
import argparse, glob, os, pickle, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir',
                   help='Path to <sweep>/work/ — looks for results/*.pkl')
    p.add_argument('--min-mb', type=int, default=10000,
                   help='Minimum peak memory in MB (default 10000 = 10 GB)')
    p.add_argument('--top', type=int, default=30)
    args = p.parse_args()

    res_dir = os.path.join(args.work_dir, 'results')
    t0 = time.time()
    n = 0
    n_success_above = 0
    rows = []
    for f in glob.glob(os.path.join(res_dir, '*.pkl')):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        n += 1
        if not r.get('success'):
            continue
        peak_kb = r.get('peak_memory_kb', 0) or 0
        peak_mb = peak_kb / 1024
        if peak_mb < args.min_mb:
            continue
        elapsed = r.get('time', 0) or 0
        steps = r.get('steps', 0) or 0
        ig = r.get('original_integral')
        base = os.path.basename(f)[:-4]   # strip .pkl
        rows.append((elapsed, peak_mb, steps, ig, base))
        n_success_above += 1
        if n % 10000 == 0:
            print(f'  [{time.time()-t0:5.1f}s] scanned {n} ...', flush=True)

    print(f'\nScanned {n} pkl files in {time.time()-t0:.1f}s')
    print(f'  success AND peak_memory >= {args.min_mb} MB : {n_success_above}')
    print()
    print(f'=== top {args.top} by SHORTEST elapsed time ===')
    print(f'{"time_s":>9s} {"peak_MB":>8s} {"steps":>5s}  integral '
          f'(basename)')
    rows.sort(key=lambda r: r[0])
    for elapsed, peak_mb, steps, ig, base in rows[:args.top]:
        ig_str = ','.join(str(x) for x in ig)
        print(f'{elapsed:>9.1f} {peak_mb:>8.0f} {steps:>5d}  I[{ig_str}]')


if __name__ == '__main__':
    main()
