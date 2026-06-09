"""Read all result.pkl files, build cache, apply substitutions, and SAVE
the result to disk. Subsequent analyses just load the saved file.

Output: <sweep_root>/replay_state.pkl  with keys:
  start_integral, prime
  cache              : dict integral -> sub-expression
  active_expr        : dict integral -> coefficient (post-substitution)
  log_integrals      : set of integrals that have at least one .log file
  built_at           : ISO timestamp
  num_pkls_scanned, n_success, n_fail
"""
import argparse, datetime, glob, os, pickle, time


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
            print(f'WARN: apply_subs hit {max_iter} iters')
            break
    print(f'  converged in {iters} iters')
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('start_integral')
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--out', default=None,
                   help='Output path (default: <sweep_root>/replay_state.pkl)')
    args = p.parse_args()

    res_dir = os.path.join(args.sweep_root, 'work', 'results')
    log_dir = os.path.join(args.sweep_root, 'work', 'logs')
    start = tuple(int(x) for x in args.start_integral.split(','))
    out_path = args.out or os.path.join(args.sweep_root, 'replay_state.pkl')

    t0 = time.time()
    print(f'[{time.time()-t0:5.1f}s] scanning {res_dir}/*.pkl ...', flush=True)
    cache = {}
    n_success = n_fail = 0
    pkls = glob.glob(os.path.join(res_dir, '*.pkl'))
    for i, f in enumerate(pkls):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        ig = r.get('original_integral')
        if ig is None:
            continue
        if r.get('success'):
            cache[ig] = r.get('final_expr', {})
            n_success += 1
        else:
            cache[ig] = {ig: 1}
            n_fail += 1
        if (i + 1) % 20000 == 0:
            print(f'[{time.time()-t0:5.1f}s]   {i+1}/{len(pkls)}', flush=True)
    print(f'[{time.time()-t0:5.1f}s] cache: {n_success} success + '
          f'{n_fail} fail-identity = {len(cache)}', flush=True)

    print(f'[{time.time()-t0:5.1f}s] applying substitutions ...', flush=True)
    active_expr = apply_subs({start: 1}, cache, args.prime)
    print(f'[{time.time()-t0:5.1f}s] active_expr: {len(active_expr)} terms',
          flush=True)

    print(f'[{time.time()-t0:5.1f}s] indexing .log files ...', flush=True)
    log_integrals = set()
    for fn in os.listdir(log_dir):
        if fn.endswith('.log'):
            ig = parse_integral(fn[:-4])
            if ig:
                log_integrals.add(ig)
    print(f'[{time.time()-t0:5.1f}s] {len(log_integrals)} integrals have a .log',
          flush=True)

    state = {
        'start_integral': start,
        'prime': args.prime,
        'cache': cache,
        'active_expr': active_expr,
        'log_integrals': log_integrals,
        'built_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'num_pkls_scanned': len(pkls),
        'n_success': n_success,
        'n_fail': n_fail,
    }

    print(f'[{time.time()-t0:5.1f}s] writing {out_path} ...', flush=True)
    with open(out_path, 'wb') as fp:
        pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)
    sz = os.path.getsize(out_path)
    print(f'[{time.time()-t0:5.1f}s] done. file size {sz/1e6:.1f} MB',
          flush=True)


if __name__ == '__main__':
    main()
