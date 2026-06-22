"""Compare two probe result.pkl's for bit-identicality.
Looks at best_state.path (full sequence of actions) and full_expr_replay
(the substituted-out final expression). Both must match exactly.

Usage:
    python compare_probe_bitident.py baseline.pkl post_fix.pkl
"""
import argparse, hashlib, pickle


def _canon_path(path):
    # path: list of (integral_tuple, op, delta_tuple)
    return repr([(tuple(t), op, tuple(d)) for t, op, d in path]).encode()


def _canon_expr(expr):
    return repr(sorted((tuple(k), v) for k, v in expr.items())).encode()


def summary(pkl_path):
    r = pickle.load(open(pkl_path, 'rb'))
    bs = r.get('best_state', {})
    path = bs.get('path', []) if isinstance(bs, dict) else []
    fer = r.get('full_expr_replay', {})
    fsr = r.get('full_subs_replay', {})
    path_h = hashlib.sha256(_canon_path(path)).hexdigest()[:16]
    fer_h = hashlib.sha256(_canon_expr(fer)).hexdigest()[:16]
    fsr_h = hashlib.sha256(repr(sorted((tuple(k), str(v))
                                        for k, v in fsr.items())).encode()
                          ).hexdigest()[:16]
    return {
        'path_len': len(path),
        'path_h': path_h,
        'expr_len': len(fer),
        'expr_h': fer_h,
        'subs_len': len(fsr),
        'subs_h': fsr_h,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('post_fix')
    args = p.parse_args()

    a = summary(args.baseline)
    b = summary(args.post_fix)

    print(f'{"field":<10s} {"baseline":<22s} {"post_fix":<22s} {"match":>6s}')
    for k in ('path_len', 'path_h', 'expr_len', 'expr_h', 'subs_len', 'subs_h'):
        m = (a[k] == b[k])
        print(f'  {k:<10s} {str(a[k]):<22s} {str(b[k]):<22s} '
              f'{"✓" if m else "✗ DIFF":>6s}')
    overall = all(a[k] == b[k] for k in a)
    print()
    print(f'BIT-IDENTICAL: {"✓ YES" if overall else "✗ NO"}')


if __name__ == '__main__':
    main()
