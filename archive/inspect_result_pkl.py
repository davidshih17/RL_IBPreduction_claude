"""Inspect a reduction result.pkl: print the actual structure so we know WHAT
each field is (and what 'masters' would even mean here) before comparing.
"""
import pickle
import sys


def show(name, v, indent='  '):
    t = type(v).__name__
    if isinstance(v, dict):
        ks = list(v.keys())
        sample_k = ks[:3]
        sample_v = [v[k] for k in sample_k]
        print(f'{indent}{name}: dict len={len(v)}')
        print(f'{indent}  sample keys : {sample_k}')
        print(f'{indent}  sample vals : {[repr(x)[:60] for x in sample_v]}')
    elif isinstance(v, (list, tuple)):
        print(f'{indent}{name}: {t} len={len(v)}; first={repr(v[0])[:80] if v else None}')
    else:
        # namedtuple / object
        print(f'{indent}{name}: {t}')
        if hasattr(v, '_fields'):
            print(f'{indent}  fields: {v._fields}')
            for f in v._fields:
                fv = getattr(v, f)
                if isinstance(fv, dict):
                    print(f'{indent}    .{f}: dict len={len(fv)}; '
                          f'sample={list(fv.items())[:2]}')
                else:
                    print(f'{indent}    .{f}: {type(fv).__name__} = {repr(fv)[:70]}')


def main():
    with open(sys.argv[1], 'rb') as f:
        obj = pickle.load(f)
    print(f'=== {sys.argv[1]} ===')
    print(f'top type: {type(obj).__name__}')
    if isinstance(obj, dict):
        print(f'top keys: {list(obj.keys())}')
        for k, v in obj.items():
            show(k, v)


if __name__ == '__main__':
    main()
