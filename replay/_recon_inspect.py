"""Recon: confirm the round1-4 combined cache is PURE-STDLIB loadable (no sailir
classes), dump its schema, and check the saved active_expr's term count.

Loads with a deliberately CLEAN sys.path (sailir dirs removed) so that if any
value were a sailir object the unpickle would fail -- which is exactly the
portability property the standalone package needs.

Run:
  python replay/_recon_inspect.py <replay_state_all4.pkl> > replay/logs/recon_inspect.log 2>&1
"""
import sys
import os
import pickle

# Strip the repo from sys.path so `import sailir` is impossible -> proves the
# pickle holds only builtins (tuple/int/dict/set/str).
REPO = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path = [p for p in sys.path if 'SAILIR_phase2' not in p and p not in ('', '.')]
if 'sailir' in sys.modules:
    del sys.modules['sailir']

path = sys.argv[1]
print(f'loading {path}  ({os.path.getsize(path)/1e6:.1f} MB)  with CLEAN sys.path')
print(f'sailir importable now? ', end='')
try:
    import sailir  # noqa
    print('YES (clean path failed)')
except Exception as e:
    print(f'NO ({type(e).__name__}) -- good, pure-stdlib load test is valid')

st = pickle.load(open(path, 'rb'))
print(f'\ntop-level keys: {sorted(st.keys())}')
print(f'start_integral: {st.get("start_integral")}')
print(f'prime:          {st.get("prime")}')

cache = st['cache']
ae = st['active_expr']
print(f'\ncache entries:       {len(cache)}')
print(f'active_expr terms:   {len(ae)}')

# sample a cache entry and verify types are pure builtins
k0 = next(iter(cache))
v0 = cache[k0]
print(f'\nsample cache key type:   {type(k0).__name__}  len={len(k0)}  -> {k0}')
print(f'sample cache val type:   {type(v0).__name__}  (#terms={len(v0)})')
if isinstance(v0, dict):
    kk = next(iter(v0))
    vv = v0[kk]
    print(f'  inner key type: {type(kk).__name__} {kk}')
    print(f'  inner val type: {type(vv).__name__} {vv}')

# scan ALL types to be 100% sure nothing exotic hides in the cache
key_types, ival_types, coeff_types = set(), set(), set()
for k, v in cache.items():
    key_types.add(type(k).__name__)
    if isinstance(v, dict):
        for ik, iv in v.items():
            ival_types.add(type(ik).__name__)
            coeff_types.add(type(iv).__name__)
print(f'\nALL cache key types:   {key_types}')
print(f'ALL inner-key types:   {ival_types}')
print(f'ALL coeff types:       {coeff_types}')

# active_expr types
ae_key_types = {type(k).__name__ for k in ae}
ae_val_types = {type(v).__name__ for v in ae.values()}
print(f'active_expr key types: {ae_key_types}')
print(f'active_expr val types: {ae_val_types}')
print('\nRECON OK' if key_types <= {'tuple'} else '\nWARNING: non-tuple keys present')
