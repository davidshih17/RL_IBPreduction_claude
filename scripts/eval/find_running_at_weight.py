"""Find currently-running condor workers whose integral matches a given
(level, r, [s]) bucket, and print the worker log file paths.

Usage:
    python find_running_at_weight.py <work_dir> --level 7 --r 7
"""
import argparse, os, re, subprocess, sys


def level_r_s(integral):
    """Return (level, r, s) for a pentagonbox 11-tuple integral.
    level = # positive among the first 8 (propagator) slots; ISP slots (8-10) excluded.
    r = sum of positive entries (clamped at 0).
    s = -sum of negative entries (clamped at 0).
    """
    ints = list(integral)
    level = sum(1 for x in ints[:8] if x > 0)
    r = sum(x for x in ints if x > 0)
    s = -sum(x for x in ints if x < 0)
    return level, r, s


def parse_int_str(s):
    return tuple(int(x) for x in s.split(','))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir')
    p.add_argument('--level', type=int, required=True)
    p.add_argument('--r', type=int, required=True)
    p.add_argument('--s', type=int, default=None)
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    log_dir = os.path.join(work_dir, 'logs')

    # condor_q just running jobs (JobStatus=2)
    cp = subprocess.run(
        ['condor_q', '-constraint', 'JobStatus==2',
         '-af', 'ClusterId', 'Args'],
        capture_output=True, text=True, timeout=60,
    )
    matches = []
    for line in cp.stdout.splitlines():
        # ClusterId<tab>Args
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue
        cid, args_str = parts
        m = re.search(r"--integral=['\"]?([\d,\-]+)", args_str)
        if not m:
            continue
        integ = parse_int_str(m.group(1))
        L, r, s = level_r_s(integ)
        if L != args.level or r != args.r:
            continue
        if args.s is not None and s != args.s:
            continue
        matches.append((cid, integ, L, r, s))

    print(f'Found {len(matches)} running worker(s) at level={args.level} r={args.r}'
          + (f' s={args.s}' if args.s is not None else '') + ':')
    for cid, integ, L, r, s in matches:
        sl = '_'.join(str(x) for x in integ)
        # Worker log naming: async_<seq>_<int_with_underscores>.out
        # We don't know the <seq> prefix from condor_q, but the basename
        # ends in _<int_with_underscores>. Glob for it.
        import glob
        log_glob = os.path.join(log_dir, f'async_*_{sl}.out')
        out_files = sorted(glob.glob(log_glob))
        print(f'  cluster {cid}: I{list(integ)} (level={L} r={r} s={s})')
        for of in out_files:
            base = of[:-4]  # strip .out
            print(f'    .out: {of}')
            for ext in ('.err', '.log'):
                fn = base + ext
                if os.path.exists(fn):
                    print(f'    {ext}: {fn}')


if __name__ == '__main__':
    main()
