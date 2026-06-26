#!/usr/bin/env python3
"""FINAL confirmation replay: clean cache (4 sanctioned pre-meta runs + meta
cascade) PLUS the 4 newly-completed standalone reductions (tgt0028 + the 3
extra* covered_by_cache fixes). Expect 996/996 ISP-clean list_TA integrals to
reduce to masters.

NOTE: the 3 extra* reduction.pkls carry the freshly-computed (clean-basis)
products that the old v6 junk used to supply -- so no v6 cache is needed."""
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


def cache_of(o):
    return o['cache'] if isinstance(o, dict) and 'cache' in o else o


def merge_pkl(combined, path):
    try:
        with open(path, 'rb') as f:
            o = pickle.load(f)
    except Exception as e:
        print(f"  SKIP {path}: {e}", flush=True); return
    c = cache_of(o)
    if isinstance(c, dict):
        combined.update(c)
    del o, c


def merge_wr(combined, results_dir):
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
            combined[integ] = r.get('final_expr', {integ: 1})


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
# clean pre-meta lineage
merge_pkl(combined, R / 'pentagonbox_8_5_v7_fresh/replay_state.pkl')
merge_wr(combined, str(R / 'pentagonbox_8_5_v7_fresh_round2/work/results'))
merge_pkl(combined, R / 'pentagonbox_7_5_cache_from_85/replay_state.pkl')
merge_pkl(combined, R / 'pentagonbox_7_6_cache_from_85_75/replay_state.pkl')
# meta cascade
burst = R / 'meta_reduce/_burst_cache/replay_state.pkl'
merge_pkl(combined, burst)
newer = subprocess.run(['find', str(R / 'meta_reduce'), '-name', 'reduction.pkl', '-newer', str(burst)],
                       capture_output=True, text=True).stdout.split()
for p in sorted(newer):
    merge_pkl(combined, p)
# the 4 newly-completed standalone reductions
for n in ['tgt0028_w7_4', 'extra0001_w6_4', 'extra0002_w6_3', 'extra0003_w6_3']:
    rp = R / f'meta_reduce/{n}/reduction.pkl'
    print(f"  + {n}/reduction.pkl exists={rp.exists()}", flush=True)
    merge_pkl(combined, rp)
print(f"\nFINAL COMBINED CACHE: {len(combined)} entries ({time.time()-t0:.0f}s)\n", flush=True)

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

print(f"========== FINAL REPLAY (ISP-clean, {len(clean)}) ==========", flush=True)
print(f"  fully reduced to masters : {fully}/{len(clean)}", flush=True)
print(f"  residual (non-masters)   : {len(residual)}/{len(clean)}", flush=True)
for t, nm in residual:
    print(f"  RESIDUAL TA[{','.join(map(str, t))}] -> {len(nm)}: {[','.join(map(str,x)) for x in nm[:6]]}", flush=True)
if not residual:
    print("\n  *** ALL 996 ISP-clean list_TA integrals reduce to masters. ***", flush=True)
