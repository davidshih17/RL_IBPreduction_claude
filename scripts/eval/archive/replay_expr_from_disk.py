"""Reconstruct the orchestrator's active expression OFFLINE from result.pkl
files on disk, with the Issue #1 fix (skip identity cache entries).

This bypasses the running orchestrator's broken apply_substitutions loop
to get the GROUND-TRUTH non-master count.

Steps:
  1. Walk work/results/*.pkl, build cache:
       success=True  -> cache[X] = final_expr
       success=False -> cache[X] = {X: 1}  (identity)
  2. apply_substitutions({start_int: 1}, cache, prime) with identity skip.
  3. Count non-masters in result, break down by:
       a. those whose cache entry is identity (TABU traps)
       b. those NOT in cache (OOM crashes — no pkl was ever written)
       c. those whose cache entry is real but somehow not substituted (bug?)

Usage:
    python replay_expr_from_disk.py <work_dir> <start_integral>
        e.g.   <start_integral> = '1,1,1,1,1,1,1,1,-5,0,0'
"""
import argparse, glob, os, pickle, sys, time


def apply_substitutions(expr, cache, prime, max_iter=200000):
    """Like the orchestrator's, but with Issue #1 fix: identity entries
    (cache[X] == {X: 1}) are treated as non-substitutions, so they don't
    keep marking changed=True forever.
    """
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
                    # Identity — no real substitution; keep X in expr unchanged
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
            print(f'  WARNING: apply_substitutions hit {max_iter} iters '
                  f'(unexpected with identity-skip fix)')
            break
    print(f'  apply_substitutions converged in {iters} iters')
    return expr


def is_master(integral, paper_masters_only=True):
    """Replicates beam_search_utils.is_master used by orchestrator.
    Paper-masters-only: integral is master iff all 8 propagators have
    weight in {0, 1} AND no negative ISP (last 3 indices)."""
    ints = list(integral)
    if paper_masters_only:
        # All 8 propagators in {0, 1}
        if any(x > 1 for x in ints[:8]):
            return False
        # No negative ISP
        if any(x < 0 for x in ints[8:]):
            return False
        return True
    # legacy: any integral with all positive indices <= 1 AND no negatives anywhere
    return all(0 <= x <= 1 for x in ints[:8]) and all(x <= 0 for x in ints[8:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir')
    p.add_argument('start_integral',
                   help='comma-separated, e.g. 1,1,1,1,1,1,1,1,-5,0,0')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    start_int = tuple(int(x) for x in args.start_integral.split(','))

    t0 = time.time()
    print(f'[{time.time()-t0:5.1f}s] scanning {work_dir}/results/*.pkl ...')
    pkls = glob.glob(os.path.join(work_dir, 'results', '*.pkl'))
    print(f'  found {len(pkls)} pkl files')

    cache = {}
    n_success = 0
    n_failed = 0
    n_unreadable = 0
    for i, f in enumerate(pkls):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            n_unreadable += 1
            continue
        ig = r.get('original_integral')
        if ig is None:
            continue
        if r.get('success'):
            cache[ig] = r.get('final_expr', {})
            n_success += 1
        else:
            cache[ig] = {ig: 1}
            n_failed += 1
        if (i + 1) % 20000 == 0:
            print(f'[{time.time()-t0:5.1f}s]   loaded {i+1}/{len(pkls)}')
    print(f'[{time.time()-t0:5.1f}s] cache built: '
          f'{n_success} success + {n_failed} fail-identity = '
          f'{len(cache)} entries  (unreadable: {n_unreadable})')

    print()
    print(f'[{time.time()-t0:5.1f}s] applying substitutions from start ...')
    expr = apply_substitutions({start_int: 1}, cache, args.prime)
    print(f'[{time.time()-t0:5.1f}s] active expr: {len(expr)} non-zero terms')

    # Classify terms in expr
    masters = []
    non_masters = []
    for ig in expr:
        if is_master(ig):
            masters.append(ig)
        else:
            non_masters.append(ig)
    print()
    print(f'  masters     : {len(masters)}')
    print(f'  non_masters : {len(non_masters)}')

    # Of the non-masters, how many are identity-cached (TABU) vs not in
    # cache (OOM crashes)?
    n_identity_cached = 0
    n_not_in_cache = 0
    n_in_cache_real = 0
    for ig in non_masters:
        if ig in cache:
            if cache[ig] == {ig: 1}:
                n_identity_cached += 1
            else:
                n_in_cache_real += 1  # would be a bug — should have substituted
        else:
            n_not_in_cache += 1
    print()
    print('  Of those non-masters:')
    print(f'    identity-cached (TABU traps)        : {n_identity_cached}')
    print(f'    NOT in cache (OOM/crash, no pkl)    : {n_not_in_cache}')
    print(f'    in cache as real expr (unexpected)  : {n_in_cache_real}')

    if n_in_cache_real:
        print()
        print('  WARNING: the "in cache as real expr" bucket should always be 0;')
        print('  if you see non-zero, that means apply_substitutions stopped')
        print('  early or there is a cache value corrupted.')


if __name__ == '__main__':
    main()
