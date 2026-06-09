"""List currently-RUNNING condor workers sorted by memory and runtime.

For each cluster in JobStatus=2 we read ClusterId, Args (to extract integral),
JobStartDate (runtime), and MemoryUsage (MB). Prints two ranked tables.

Usage:
    python find_hog_workers.py [--top 30] [--min-mb 4000] [--min-h 1]
"""
import argparse, re, subprocess, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=30)
    p.add_argument('--min-mb', type=int, default=0,
                   help='Only show workers above this MemoryUsage (MB).')
    p.add_argument('--min-h', type=float, default=0,
                   help='Only show workers running longer than this (hours).')
    args = p.parse_args()

    now = int(time.time())

    cp = subprocess.run(
        ['condor_q', '-constraint', 'JobStatus==2',
         '-af', 'ClusterId', 'Args', 'JobStartDate', 'MemoryUsage'],
        capture_output=True, text=True, timeout=120,
    )

    rows = []
    n_undef_mem = 0
    n_undef_start = 0
    for line in cp.stdout.splitlines():
        # Format: "ClusterId Args JobStartDate MemoryUsage". Args has spaces,
        # so split off the last 2 tokens from the right.
        parts = line.rsplit(None, 2)
        if len(parts) < 3:
            continue
        mem_tok = parts[-1]
        start_tok = parts[-2]
        # MemoryUsage / JobStartDate can be "undefined" for very-fresh jobs
        # the schedd has not polled yet — treat as 0 so they don't disappear
        # from the listing entirely, but tag the runtime.
        try:
            mem_mb = int(mem_tok)
        except ValueError:
            try:
                mem_mb = int(float(mem_tok))
            except ValueError:
                mem_mb = 0
                n_undef_mem += 1
        try:
            start = int(start_tok)
            runtime_h = (now - start) / 3600.0
        except ValueError:
            start = 0
            runtime_h = 0.0
            n_undef_start += 1
        head = parts[0]
        head_parts = head.split(None, 1)
        if len(head_parts) != 2:
            continue
        cid, args_str = head_parts
        m = re.search(r"--integral=['\"]?([\d,\-]+)", args_str)
        if not m:
            continue
        integ = m.group(1)
        if mem_mb < args.min_mb:
            continue
        if runtime_h < args.min_h:
            continue
        rows.append((cid, integ, runtime_h, mem_mb))

    print(f'Total running workers matching filter: {len(rows)}')
    if n_undef_mem or n_undef_start:
        print(f'  (warning: {n_undef_mem} undefined MemoryUsage,'
              f' {n_undef_start} undefined JobStartDate — schedd not yet polled)')

    print()
    print(f'=== TOP {args.top} by MemoryUsage (MB) ===')
    print(f'{"cluster":>10s} {"runtime_h":>9s} {"mem_MB":>8s}  integral')
    for cid, integ, rh, mb in sorted(rows, key=lambda r: -r[3])[:args.top]:
        print(f'{cid:>10s} {rh:>9.2f} {mb:>8d}  I[{integ}]')

    print()
    print(f'=== TOP {args.top} by runtime (hours) ===')
    print(f'{"cluster":>10s} {"runtime_h":>9s} {"mem_MB":>8s}  integral')
    for cid, integ, rh, mb in sorted(rows, key=lambda r: -r[2])[:args.top]:
        print(f'{cid:>10s} {rh:>9.2f} {mb:>8d}  I[{integ}]')


if __name__ == '__main__':
    main()
