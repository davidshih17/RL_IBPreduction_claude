"""Find currently-running condor workers whose target integral is NO LONGER
in the orchestrator's active expression — exact (per-integral) check.

We reconstruct the orchestrator's `expr` offline by scanning every result.pkl
under <work_dir>/results/, building the cache, and applying
apply_substitutions({start_int: 1}, cache, prime) — same logic as
hierarchical_reduction.py:apply_substitutions. Failed-worker pickles are
cached as identity (matching the orchestrator's behaviour).

Then for each currently-running worker, we check whether its integral is
in the offline-reconstructed expr.

Usage:
    python find_zombie_workers_exact.py <work_dir> <start_integral>
        e.g.   <start_integral> = '1,1,1,1,1,1,1,1,-5,0,0'
"""
import argparse, glob, os, pickle, re, subprocess
from collections import defaultdict


def apply_substitutions(expr, cache, prime, max_iter=100_000):
    """Replay of hierarchical_reduction.py:apply_substitutions."""
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
                # if cache[integ] == {integ: 1}, this is identity → skip
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
            print(f'WARNING: apply_substitutions hit {max_iter} iters; '
                  f'returning partially converged expr')
            break
    return expr


def parse_int_str(s):
    return tuple(int(x) for x in s.split(','))


def level_r_s(integral):
    ints = list(integral)
    L = sum(1 for x in ints[:8] if x > 0)
    r = sum(x for x in ints if x > 0)
    s = -sum(x for x in ints if x < 0)
    return L, r, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir')
    p.add_argument('start_integral',
                   help='Start integral as comma-separated, e.g. 1,1,1,1,1,1,1,1,-5,0,0')
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--level', type=int, default=None,
                   help='Filter output to this level only.')
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    start_int = parse_int_str(args.start_integral)

    # Build cache from result.pkl files (identity for failed workers).
    print(f'Building cache from {work_dir}/results/...')
    cache = {}
    n_ok = n_fail = 0
    for f in glob.glob(os.path.join(work_dir, 'results', '*.pkl')):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        i = r.get('original_integral')
        if i is None:
            continue
        if r.get('success'):
            cache[i] = r.get('final_expr', {})
            n_ok += 1
        else:
            cache[i] = {i: 1}
            n_fail += 1
    print(f'  cache: {n_ok} success + {n_fail} fail-identity = {len(cache)} entries')

    # Reconstruct expr.
    print('Reconstructing active expr offline...')
    expr = apply_substitutions({start_int: 1}, cache, args.prime)
    print(f'  active expr: {len(expr)} non-zero terms')

    # Find currently-running workers + their integrals.
    print('Querying running condor workers...')
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
        running.append((cid, integ))
    print(f'  running workers: {len(running)}')

    # Classify each running worker.
    zombies = []
    still_needed = []
    for cid, integ in running:
        L, r, s = level_r_s(integ)
        if args.level is not None and L != args.level:
            continue
        if integ in expr:
            still_needed.append((cid, integ, L, r, s))
        else:
            zombies.append((cid, integ, L, r, s))

    title = (f'level={args.level}' if args.level is not None else 'ALL levels')
    print()
    print(f'== Running workers ({title}) ==')
    print(f'  still needed: {len(still_needed)}')
    print(f'  ZOMBIES (integral NOT in current expr): {len(zombies)}')

    if zombies:
        # Bucket the zombies by (L, r, s)
        buckets = defaultdict(int)
        for cid, integ, L, r, s in zombies:
            buckets[(L, r, s)] += 1
        print()
        print('Zombies by (L, r, s) bucket:')
        for k in sorted(buckets):
            print(f'  L={k[0]} r={k[1]} s={k[2]}:  {buckets[k]} worker(s)')
        print()
        if args.level is not None:
            print(f'Per-cluster list (level {args.level}):')
            for cid, integ, L, r, s in zombies:
                print(f'  cluster {cid}: I{list(integ)} (L={L} r={r} s={s})')


if __name__ == '__main__':
    main()
