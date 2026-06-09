"""Replay active expression from disk and tabulate non-masters by (L, r, s).
Also splits the table by failure category so you can see what's stuck where.

Categories applied per non-master:
  - TABU      : in cache as identity {X: 1}
  - DISPATCHED_DIED : not in cache, has .log file
  - NEVER     : not in cache, no .log file
"""
import argparse, glob, os, pickle, time
from collections import defaultdict, Counter


def apply_subs(expr, cache, prime, max_iter=200000):
    changed = True
    iters = 0
    while changed:
        changed = False
        iters += 1
        new_expr = {}
        for ig, coeff in expr.items():
            if coeff == 0:
                continue
            if ig in cache:
                v = cache[ig]
                if v == {ig: 1}:
                    new_expr[ig] = (new_expr.get(ig, 0) + coeff) % prime
                    continue
                for k, c in v.items():
                    if c == 0:
                        continue
                    new_expr[k] = (new_expr.get(k, 0) + coeff * c) % prime
                changed = True
            else:
                new_expr[ig] = (new_expr.get(ig, 0) + coeff) % prime
        expr = {k: v for k, v in new_expr.items() if v != 0}
        if iters > max_iter:
            break
    return expr


def parse_integral(base):
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


def Lrs(ig):
    L = sum(1 for x in ig[:8] if x > 0)
    r = sum(x for x in ig if x > 0)
    s = -sum(x for x in ig if x < 0)
    return L, r, s


def is_paper_master(ig):
    return all(0 <= x <= 1 for x in ig[:8]) and all(x >= 0 for x in ig[8:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('start_integral')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    sw = args.sweep_root
    res_dir = os.path.join(sw, 'work/results')
    log_dir = os.path.join(sw, 'work/logs')
    start = tuple(int(x) for x in args.start_integral.split(','))

    t0 = time.time()
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
    print(f'  {len(cache)} cache entries in {time.time()-t0:.1f}s', flush=True)

    expr = apply_subs({start: 1}, cache, args.prime)

    # Index .log files
    log_set = set()
    for fn in os.listdir(log_dir):
        if fn.endswith('.log'):
            ig = parse_integral(fn[:-4])
            if ig:
                log_set.add(ig)

    # Classify and bucket
    by_Lrs = defaultdict(Counter)   # (L,r,s) -> Counter(cat)
    cat_totals = Counter()
    L_totals = Counter()
    for ig in expr:
        if is_paper_master(ig):
            continue  # only non-masters
        L, r, s = Lrs(ig)
        if ig in cache:
            cat = 'TABU' if cache[ig] == {ig: 1} else 'CACHED_REAL'  # shouldn't happen
        elif ig in log_set:
            cat = 'DISPATCHED_DIED'
        else:
            cat = 'NEVER'
        by_Lrs[(L, r, s)][cat] += 1
        cat_totals[cat] += 1
        L_totals[L] += 1

    print()
    print(f'total non-masters in active expr: {sum(cat_totals.values())}')
    print()
    print('=== by category ===')
    for k in sorted(cat_totals, key=lambda c: -cat_totals[c]):
        print(f'  {k:<18s} {cat_totals[k]}')
    print()
    print('=== by level L ===')
    for L in sorted(L_totals, key=lambda x: -x):
        print(f'  L={L:<2d} {L_totals[L]:>5d}')
    print()
    print('=== by (L, r, s) — full table ===')
    print(f'{"L":>3s} {"r":>3s} {"s":>3s} {"total":>6s} {"TABU":>6s} '
          f'{"DIED":>6s} {"NEVER":>6s}')
    for k in sorted(by_Lrs, key=lambda t: (-t[0], -t[1], -t[2])):
        L, r, s = k
        c = by_Lrs[k]
        tot = sum(c.values())
        print(f'{L:>3d} {r:>3d} {s:>3d} {tot:>6d} '
              f'{c.get("TABU",0):>6d} {c.get("DISPATCHED_DIED",0):>6d} '
              f'{c.get("NEVER",0):>6d}')


if __name__ == '__main__':
    main()
