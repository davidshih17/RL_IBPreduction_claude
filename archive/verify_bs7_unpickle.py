#!/usr/bin/env python3
"""Verify the onestep_worker_v7 path fix: with SAILIR_phase2 + scripts/eval on
sys.path, an existing beam_search_v7 *.bs7.pkl must unpickle and expose the
fields the worker translates (best_state.path, full_expr_replay). Prints PASS/FAIL.
"""
import glob
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # .../SAILIR_phase2
sys.path.insert(0, str(_HERE))                  # .../scripts/eval

RES = (_HERE.parent.parent /
       'results/pentagonbox_8_5_v6_round3/work/results')
files = sorted(glob.glob(str(RES / '*.bs7.pkl')))
print(f'found {len(files)} .bs7.pkl files', flush=True)
if not files:
    print('NO FILES TO TEST', flush=True)
    sys.exit(2)

f = files[0]
print(f'loading: {f}', flush=True)
with open(f, 'rb') as fh:
    bs7 = pickle.load(fh)
print('UNPICKLE OK. keys:', sorted(bs7.keys()), flush=True)
bstate = bs7.get('best_state', {})
path = list(bstate.get('path', []))
fe = bs7.get('full_expr_replay')
print(f'best_state.path len = {len(path)}', flush=True)
print(f'full_expr_replay terms = {None if fe is None else len(fe)}', flush=True)
ok = (len(path) > 0) and (fe is not None)
print('PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
