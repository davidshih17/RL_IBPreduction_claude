"""Compare two reduction result.pkl files: are the final reductions identical?

A correct IBP reduction is unique given the master basis, so two different
search PATHS should still yield the SAME final expression (same master
integrals AND same coefficients). This checks that — the real correctness bar,
stronger than path-length equality.

Usage: python compare_reduction_results.py <baseline.pkl> <candidate.pkl>
"""
import os
import pickle
import sys

# Make `sailir` + the beam_search modules importable for unpickling result.pkl
# (it holds State_v5 namedtuples and sailir-referenced objects).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts', 'eval'))


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def summarize(obj):
    """Return a dict describing the result, robust to the pkl schema."""
    info = {'type': type(obj).__name__}
    if isinstance(obj, dict):
        info['keys'] = list(obj.keys())
    return info


def get_final_expr(obj):
    """Extract the final reduced expression (master_integral -> coeff)."""
    if isinstance(obj, dict):
        for k in ('final_expr', 'expr', 'reduced', 'result', 'final'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # nested: {start_int: final_expr_dict}
        for v in obj.values():
            if isinstance(v, dict) and v and all(isinstance(kk, tuple) for kk in v):
                return v
    return None


def main():
    base_path, cand_path = sys.argv[1], sys.argv[2]
    base, cand = load(base_path), load(cand_path)
    print(f'baseline : {base_path}')
    print(f'  {summarize(base)}')
    print(f'candidate: {cand_path}')
    print(f'  {summarize(cand)}')
    print()

    be, ce = get_final_expr(base), get_final_expr(cand)
    if be is None or ce is None:
        print('COULD NOT extract final_expr from one/both — dumping top-level repr:')
        print('  baseline :', repr(base)[:500])
        print('  candidate:', repr(cand)[:500])
        return

    bk, ck = set(be.keys()), set(ce.keys())
    print(f'baseline  masters (n={len(bk)})')
    print(f'candidate masters (n={len(ck)})')
    print(f'  masters IDENTICAL set: {bk == ck}')
    if bk != ck:
        print(f'  only in baseline : {sorted(bk - ck)[:10]}')
        print(f'  only in candidate: {sorted(ck - bk)[:10]}')

    # Coefficient comparison over the shared keys.
    common = bk & ck
    coeff_mismatch = [k for k in common if be[k] != ce[k]]
    print(f'  coeff mismatches over {len(common)} shared masters: {len(coeff_mismatch)}')
    if coeff_mismatch[:5]:
        for k in coeff_mismatch[:5]:
            print(f'    {k}: baseline={be[k]}  candidate={ce[k]}')

    identical = (bk == ck) and not coeff_mismatch
    print()
    print(f'==> FINAL REDUCTION IDENTICAL: {identical}')


if __name__ == '__main__':
    main()
