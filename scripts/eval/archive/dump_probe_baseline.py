"""Dump the path+key-stats of each existing probe's result.pkl so we can
diff against the post-fix re-run.
"""
import pickle, sys, hashlib


def summarize(pkl_path):
    try:
        r = pickle.load(open(pkl_path, 'rb'))
    except Exception as e:
        return f'UNREADABLE: {e}'
    success = r.get('success')
    steps = r.get('steps', 0)
    time_s = r.get('time', 0)
    nm = r.get('best_n_non_masters', '-')
    mw = r.get('best_max_w12', '-')
    peak_mb = (r.get('peak_memory_kb', 0) or 0) / 1024
    path = r.get('path', [])
    final_expr = r.get('final_expr', {}) or {}
    # Hash the path (as canonical bytes) — bit-identicality means same path
    p_canonical = repr([(tuple(t), op, tuple(d)) for t, op, d in path]).encode()
    path_hash = hashlib.sha256(p_canonical).hexdigest()[:16]
    # Hash the final_expr (sorted keys)
    fe_canonical = repr(sorted((tuple(k), v) for k, v in final_expr.items())).encode()
    fe_hash = hashlib.sha256(fe_canonical).hexdigest()[:16]
    return (f'success={success} steps={steps} time={time_s:.0f}s '
            f'best_nm={nm} best_mw={mw} peak_MB={peak_mb:.0f} '
            f'path_len={len(path)} path_hash={path_hash} '
            f'final_expr_len={len(final_expr)} fe_hash={fe_hash}')


for pkl in sys.argv[1:]:
    print(f'{pkl}\n  {summarize(pkl)}')
