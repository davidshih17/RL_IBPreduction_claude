"""For the active expr's not-in-cache integrals that have NO .log file,
dump samples and check whether they are masters (no worker needed) or
non-masters (orchestrator should have dispatched). Doesn't infer cause.
"""
import argparse, glob, os, pickle, time
from collections import Counter


def apply_substitutions_with_skip(expr, cache, prime, max_iter=200000):
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
            break
    return expr


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
    return tuple(int(x) for x in nums[:11][::-1])


def level_r_s(integral):
    L = sum(1 for x in integral[:8] if x > 0)
    r = sum(x for x in integral if x > 0)
    s = -sum(x for x in integral if x < 0)
    return L, r, s


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

    # build cache
    print('reading pkls ...', flush=True)
    cache = {}
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
    print(f'  cache {len(cache)} entries in {time.time()-t0:.1f}s', flush=True)

    expr = apply_substitutions_with_skip({start_int: 1}, cache, args.prime)

    # index logs by integral
    log_integ = set()
    for fn in os.listdir(log_dir):
        if not fn.endswith('.log'):
            continue
        ig = parse_integral_from_basename(fn[:-4])
        if ig is not None:
            log_integ.add(ig)

    # Collect "never_submitted" = in expr, not in cache, no .log file
    never = [ig for ig in expr if (ig not in cache) and (ig not in log_integ)]
    print(f'\nactive expr terms: {len(expr)}')
    print(f'  with cache entry           : {sum(1 for ig in expr if ig in cache)}')
    print(f'  no cache, with .log        : {sum(1 for ig in expr if ig not in cache and ig in log_integ)}')
    print(f'  no cache, no .log (NEVER)  : {len(never)}')
    print()

    # For each NEVER, check what its (L,r,s) is and whether it has any
    # positive index at position 8-10 (ISP — paper-master criterion fails)
    print('=== NEVER_SUBMITTED breakdown ===')
    by_L = Counter()
    by_paper_master = Counter()  # paper-master criterion
    by_all_positive_le_1 = Counter()  # all 8 propagators in {0,1} AND ISP in {0}
    samples = []
    for ig in never:
        L, r, s = level_r_s(ig)
        by_L[L] += 1
        # paper-master criterion (used by --paper-masters-only in code):
        # all 8 propagators in {0, 1} AND no negative ISP (last 3 indices)
        is_paper_master = (all(0 <= x <= 1 for x in ig[:8])
                           and all(x >= 0 for x in ig[8:]))
        by_paper_master['paper_master' if is_paper_master else 'non_master'] += 1
        # legacy master: all in {0,1} and last 3 <= 0
        is_legacy_master = (all(0 <= x <= 1 for x in ig[:8])
                            and all(x <= 0 for x in ig[8:]))
        by_all_positive_le_1['legacy_master' if is_legacy_master
                              else 'not_legacy_master'] += 1
        if len(samples) < 10:
            samples.append(ig)

    print(f'by level L:')
    for k in sorted(by_L, key=lambda x: -x):
        print(f'  L={k:<2d} {by_L[k]:>5d}')
    print()
    print('by paper_master criterion '
          '(all 8 props in {0,1} AND no negative ISP):')
    for k, v in by_paper_master.items():
        print(f'  {k:<14s} {v:>5d}')
    print()
    print('by legacy_master criterion '
          '(all 8 in {0,1} AND last 3 <= 0):')
    for k, v in by_all_positive_le_1.items():
        print(f'  {k:<14s} {v:>5d}')
    print()
    print('=== 10 sample NEVER_SUBMITTED integrals ===')
    for ig in samples:
        L, r, s = level_r_s(ig)
        print(f'  I{list(ig)}  L={L} r={r} s={s}')


if __name__ == '__main__':
    main()
