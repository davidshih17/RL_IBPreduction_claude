"""Kill currently-RUNNING workers above a memory threshold and record the
integrals to a persistent manual-OOM list. Re-queries condor at kill time
so we never act on stale data.

Records:
    <sweep>/manual_oom_killed.csv
    columns: cluster_id, integral, mem_mb, runtime_h, kill_time_iso,
             worker_basename

Usage:
    python kill_hog_workers.py <sweep_root> --threshold-mb 8000 [--dry-run]
"""
import argparse, datetime, os, re, subprocess, sys, time


def query_running():
    cp = subprocess.run(
        ['condor_q', '-constraint', 'JobStatus==2',
         '-af', 'ClusterId', 'Args', 'JobStartDate', 'MemoryUsage'],
        capture_output=True, text=True, timeout=120,
    )
    if cp.returncode != 0:
        print(f'condor_q failed: {cp.stderr}', file=sys.stderr)
        sys.exit(2)
    return cp.stdout.splitlines()


def parse_row(line, now):
    parts = line.rsplit(None, 2)
    if len(parts) < 3:
        return None
    try:
        mem_mb = int(parts[-1])
    except ValueError:
        return None
    try:
        start = int(parts[-2])
    except ValueError:
        return None
    head = parts[0]
    head_parts = head.split(None, 1)
    if len(head_parts) != 2:
        return None
    cid, args_str = head_parts
    m_int = re.search(r"--integral=['\"]?([\d,\-]+)", args_str)
    m_out = re.search(r"--output\s+(\S+)", args_str)
    if not m_int:
        return None
    integ = m_int.group(1)
    runtime_h = (now - start) / 3600.0
    base = os.path.basename(m_out.group(1)) if m_out else ''
    if base.endswith('.pkl'):
        base = base[:-4]
    return (cid, integ, runtime_h, mem_mb, base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('--threshold-mb', type=int, default=8000)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    out_csv = os.path.join(args.sweep_root, 'manual_oom_killed.csv')
    write_header = not os.path.exists(out_csv)

    now = int(time.time())
    rows = []
    for line in query_running():
        r = parse_row(line, now)
        if r is None:
            continue
        if r[3] >= args.threshold_mb:
            rows.append(r)

    rows.sort(key=lambda r: -r[3])  # biggest first
    print(f'Found {len(rows)} running workers >= {args.threshold_mb} MB')
    if not rows:
        return

    print()
    print(f'{"cluster":>10s} {"runtime_h":>9s} {"mem_MB":>8s}  integral')
    for cid, integ, rh, mb, _ in rows:
        print(f'{cid:>10s} {rh:>9.2f} {mb:>8d}  I[{integ}]')

    if args.dry_run:
        print('\n[DRY-RUN] no kills issued')
        return

    print(f'\nIssuing condor_rm for {len(rows)} specific cluster IDs ...')
    cids = [r[0] for r in rows]
    # Kill in chunks of 50 to avoid argv length issues on huge lists.
    n_killed = 0
    for i in range(0, len(cids), 50):
        chunk = cids[i:i+50]
        cp = subprocess.run(['condor_rm'] + chunk,
                            capture_output=True, text=True, timeout=60)
        if cp.returncode == 0:
            n_killed += len(chunk)
            for cid in chunk:
                print(f'  condor_rm {cid}: ok')
        else:
            print(f'  condor_rm chunk failed (returncode={cp.returncode}): '
                  f'{cp.stderr.strip()}')
            # try one-by-one fallback
            for cid in chunk:
                cp1 = subprocess.run(['condor_rm', cid],
                                     capture_output=True, text=True,
                                     timeout=20)
                ok = (cp1.returncode == 0)
                print(f'  condor_rm {cid}: {"ok" if ok else "FAIL "+cp1.stderr.strip()}')
                if ok:
                    n_killed += 1

    print(f'\nKilled {n_killed} / {len(rows)} requested workers')

    iso = datetime.datetime.now().isoformat(timespec='seconds')
    with open(out_csv, 'a') as fp:
        if write_header:
            fp.write('cluster_id,integral,mem_mb,runtime_h,kill_time_iso,'
                     'worker_basename\n')
        for cid, integ, rh, mb, base in rows:
            integ_q = integ.replace(',', ';')
            fp.write(f'{cid},{integ_q},{mb},{rh:.2f},{iso},{base}\n')
    print(f'Recorded to {out_csv}')


if __name__ == '__main__':
    main()
