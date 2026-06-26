#!/usr/bin/env python3
"""CLEAN replay: combined cache = meta cascade + ONLY the 4 sanctioned pre-meta
runs (the v7_fresh lineage). Explicitly EXCLUDES v6*/v3/v1_buggy/first/second/
85_75_combined (old/buggy/different-basis).

Sanctioned pre-meta runs:
  pentagonbox_8_5_v7_fresh
  pentagonbox_8_5_v7_fresh_round2   (no replay_state -> work/results)
  pentagonbox_7_5_cache_from_85
  pentagonbox_7_6_cache_from_85_75
Meta cascade: _burst_cache + _resume_cache + the 27 reduction.pkls newer than
_burst_cache + tgt0028 live work/results.

Diagnostic: for the known residual keys, report which source (if any) carries
them -- in particular to confirm the meta caches are NOT contaminated with the
v6-only '-4' reductions.
"""
import pickle, re, sys, glob, os, subprocess, time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'reduction'))
from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, is_master

PRIME = 1009
init_from_topology(Topology.from_dir(str(BASE / 'topology_input/pentagonbox')))
set_prime(PRIME); set_paper_masters_only(False)
print(f"topology=pentagonbox prime={PRIME} paper_masters_only=False\n", flush=True)

DIAG = {
    'gap 2,1,1,0,-2,1,0,1,0,0,0':      (2, 1, 1, 0, -2, 1, 0, 1, 0, 0, 0),
    'v6sub 1,0,2,1,0,2,0,0,-4,0,0':    (1, 0, 2, 1, 0, 2, 0, 0, -4, 0, 0),
    'v6sub 2,0,1,1,0,2,0,0,-4,0,0':    (2, 0, 1, 1, 0, 2, 0, 0, -4, 0, 0),
    'v6sub 1,0,1,2,0,2,0,0,-4,0,0':    (1, 0, 1, 2, 0, 2, 0, 0, -4, 0, 0),
    'v6sub 1,1,1,0,-1,1,0,2,0,-2,0':   (1, 1, 1, 0, -1, 1, 0, 2, 0, -2, 0),
}


def cache_of(o):
    return o['cache'] if isinstance(o, dict) and 'cache' in o else o


def diag_line(label, c):
    hits = [name for name, k in DIAG.items() if k in c]
    return f"  {label:42s} |cache|={len(c):>7}  carries: {hits if hits else '-'}"


def merge_pkl(combined, path, label):
    try:
        with open(path, 'rb') as f:
            o = pickle.load(f)
    except Exception as e:
        print(f"  SKIP {label}: {e}", flush=True); return
    c = cache_of(o)
    if isinstance(c, dict):
        print(diag_line(label, c), flush=True)
        combined.update(c)
    del o, c


def merge_workresults(combined, results_dir, label):
    local = {}
    for f in glob.glob(os.path.join(results_dir, 'async_*.pkl')):
        if os.path.getsize(f) == 0:
            continue
        try:
            with open(f, 'rb') as fh:
                r = pickle.load(fh)
        except Exception:
            continue
        integ = r.get('original_integral')
        if integ is not None and r.get('success'):
            local[integ] = r.get('final_expr', {integ: 1})
    print(diag_line(label, local), flush=True)
    combined.update(local)


def load_targets(p):
    out = []
    for ln in open(p):
        m = re.match(r'TA\[([0-9,\-]+)\]', ln.strip())
        if m:
            out.append(tuple(int(x) for x in m.group(1).split(',')))
    return out


def apply_substitutions(expr, cache, prime):
    changed, it = True, 0
    while changed:
        changed, it = False, it + 1
        new = {}
        for integ, coeff in expr.items():
            if coeff == 0:
                continue
            sub = cache.get(integ)
            if sub is not None:
                for si, sc in sub.items():
                    if sc:
                        new[si] = (new.get(si, 0) + coeff * sc) % prime
                changed = True
            else:
                new[integ] = (new.get(integ, 0) + coeff) % prime
        expr = {k: v for k, v in new.items() if v != 0}
        if it > 10000:
            break
    return expr


t0 = time.time()
combined = {}
R = BASE / 'results'
print("=== sanctioned pre-meta runs (v7_fresh lineage) ===", flush=True)
merge_pkl(combined, R / 'pentagonbox_8_5_v7_fresh/replay_state.pkl', 'pentagonbox_8_5_v7_fresh')
merge_workresults(combined, str(R / 'pentagonbox_8_5_v7_fresh_round2/work/results'), 'pentagonbox_8_5_v7_fresh_round2 (wr)')
merge_pkl(combined, R / 'pentagonbox_7_5_cache_from_85/replay_state.pkl', 'pentagonbox_7_5_cache_from_85')
merge_pkl(combined, R / 'pentagonbox_7_6_cache_from_85_75/replay_state.pkl', 'pentagonbox_7_6_cache_from_85_75')

print("\n=== meta cascade caches (check these are v6-free) ===", flush=True)
burst = R / 'meta_reduce/_burst_cache/replay_state.pkl'
merge_pkl(combined, burst, '_burst_cache')
# NOTE: _resume_cache intentionally EXCLUDED (built this session by the finisher;
# unverified provenance). The 27 newer reduction.pkls cover everything finished
# after _burst_cache anyway.
newer = subprocess.run(['find', str(R / 'meta_reduce'), '-name', 'reduction.pkl', '-newer', str(burst)],
                       capture_output=True, text=True).stdout.split()
for p in sorted(newer):
    merge_pkl(combined, p, 'newer:' + Path(p).parent.name)
merge_workresults(combined, str(R / 'meta_reduce/tgt0028_w7_4/work/results'), 'tgt0028 (wr)')

print(f"\nCLEAN COMBINED CACHE: {len(combined)} entries ({time.time()-t0:.0f}s)\n", flush=True)

clean = load_targets(BASE / 'from_federica/list_TA_ispclean_by_weight')
print(f"replaying {len(clean)} ISP-clean list_TA integrals ...\n", flush=True)
fully = 0
residual = []
for t in clean:
    expr = apply_substitutions({t: 1}, combined, PRIME)
    nm = [i for i in expr if not is_master(i)]
    if nm:
        residual.append((t, nm))
    else:
        fully += 1

print(f"========== CLEAN REPLAY (ISP-clean, {len(clean)}) ==========", flush=True)
print(f"  fully reduced to masters : {fully}/{len(clean)}", flush=True)
print(f"  residual (non-masters)   : {len(residual)}/{len(clean)}", flush=True)
for t, nm in residual:
    print(f"\n  RESIDUAL  TA[{','.join(map(str, t))}]  -> {len(nm)} non-master term(s):", flush=True)
    for x in nm[:15]:
        print(f"      {','.join(map(str, x))}", flush=True)
