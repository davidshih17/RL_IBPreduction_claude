"""Find currently-running condor workers whose target integral is NO LONGER
in the orchestrator's active expression (i.e. zombie workers — substituted out
via cache chains during their long run; their result.pkl will arrive but be
useless for the global reduction).

We approximate "active expr" by parsing the orchestrator log's MOST RECENT
[hist] histogram + [maxw] line. An integral is "no longer needed" if its
(level, r) bucket is missing from the histogram, OR if its s is greater
than the bucket's printed max-s.

This is approximate — the histogram doesn't print full integral lists. So we
do a conservative check: filter currently-running workers to those whose
weight bucket is NOT in the histogram at all.

Usage:
    python find_zombie_running_workers.py <work_dir> --level 7
"""
import argparse, os, re, subprocess


def level_r_s(integral):
    ints = list(integral)
    L = sum(1 for x in ints[:8] if x > 0)
    r = sum(x for x in ints if x > 0)
    s = -sum(x for x in ints if x < 0)
    return L, r, s


def parse_int_str(s):
    return tuple(int(x) for x in s.split(','))


def parse_hist_line(line):
    # `[hist] L7r7:5 L6r8:65 ...` → set of (level, r) pairs present
    parts = re.findall(r'L(\d+)r(\d+):\d+', line)
    return {(int(L), int(r)) for L, r in parts}


def parse_maxw_line(line):
    # `[maxw] L7: n=5 max=(r=7,s=4) L6: ...` → {L: (max_r, max_s)}
    pairs = re.findall(r'L(\d+):\s*n=\d+\s*max=\(r=(\d+),s=(\d+)\)', line)
    return {int(L): (int(r), int(s)) for L, r, s in pairs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir')
    p.add_argument('--level', type=int, default=None,
                   help='Filter to one level (e.g. 7). Default: all levels.')
    p.add_argument('--orch-log', default=None,
                   help='Orchestrator log path; default derived from work_dir.')
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if args.orch_log is None:
        # work_dir = .../<exp>/work; orch log lives at .../<exp>/logs/hierarchical.log
        exp_root = os.path.dirname(work_dir)
        orch_log = os.path.join(exp_root, 'logs', 'hierarchical.log')
    else:
        orch_log = args.orch_log

    # Tail to get latest [hist] and [maxw]
    last_hist = last_maxw = None
    with open(orch_log) as f:
        for line in f:
            if '[hist]' in line:
                last_hist = line.strip()
            elif '[maxw]' in line:
                last_maxw = line.strip()
    if last_hist is None:
        print('no [hist] line found in orchestrator log')
        return
    active_buckets = parse_hist_line(last_hist)
    maxw = parse_maxw_line(last_maxw) if last_maxw else {}
    print(f'Latest [hist]: {last_hist}')
    print(f'Latest [maxw]: {last_maxw}')
    print(f'Active (L,r) buckets in expr: {len(active_buckets)}')
    print()

    # Get running workers' integrals
    cp = subprocess.run(
        ['condor_q', '-constraint', 'JobStatus==2',
         '-af', 'ClusterId', 'Args'],
        capture_output=True, text=True, timeout=60,
    )
    running = []
    for line in cp.stdout.splitlines():
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue
        cid, args_str = parts
        m = re.search(r"--integral=['\"]?([\d,\-]+)", args_str)
        if not m:
            continue
        integ = parse_int_str(m.group(1))
        L, r, s = level_r_s(integ)
        running.append((cid, integ, L, r, s))

    zombies = []
    active_running = []
    for cid, integ, L, r, s in running:
        if args.level is not None and L != args.level:
            continue
        bucket_present = (L, r) in active_buckets
        # Also check if s exceeds the L's max-s (likely zombie)
        s_exceeds = False
        if L in maxw:
            max_r, max_s = maxw[L]
            if r < max_r or (r == max_r and s > max_s):
                # outside the active frontier for this level
                # (lower r means likely substituted out)
                pass
        if not bucket_present:
            zombies.append((cid, integ, L, r, s))
        else:
            active_running.append((cid, integ, L, r, s))

    title = (f'level={args.level}' if args.level is not None else 'ALL levels')
    print(f'== Running workers ({title}) ==')
    print(f'  TOTAL running: {len(running) if args.level is None else len([x for x in running if x[2]==args.level])}')
    print(f'  Probable ZOMBIES (bucket not in current [hist]): {len(zombies)}')
    print(f'  Still NEEDED (bucket in [hist]): {len(active_running)}')

    if zombies:
        print()
        print(f'Zombie workers:')
        for cid, integ, L, r, s in zombies:
            print(f'  cluster {cid}: I{list(integ)} (L={L} r={r} s={s})')


if __name__ == '__main__':
    main()
