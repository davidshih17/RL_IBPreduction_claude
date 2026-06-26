#!/usr/bin/env python3
"""COMPLETE replay: confirm each ISP-clean list_TA integral reduces to masters
using EVERY reduction we have -- the meta cascade AND the pre-meta runs (the
monster (8,5) v6/v6_round3/v6_round4, v7_fresh, and the 7_5 / 7_6 / 85_75
cache-reuse runs), plus tgt0028's live work/results.

Combined cache sources (unioned; meta merged last so it wins on conflicts):
  - pre-meta:  every results/<run>/replay_state.pkl outside meta_reduce/
               + work/results of any pre-meta run dir lacking a replay_state
                 (e.g. v3 / delta)
  - meta:      _resume_cache + _burst_cache + the 27 reduction.pkls newer than
               _burst_cache
  - tgt0028:   live work/results
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
print(f"topology=pentagonbox prime={PRIME} paper_masters_only=False", flush=True)


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


def merge_workresults(combined, results_dir):
    n = 0
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
            n += 1
    return n


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

# --- pre-meta replay_states ---
premeta_rs = sorted(p for p in glob.glob(str(BASE / 'results/*/replay_state.pkl'))
                    if 'meta_reduce' not in p)
print(f"[pre-meta] {len(premeta_rs)} replay_state.pkl:", flush=True)
for p in premeta_rs:
    merge_pkl(combined, p)
    print(f"  merged {p.split('/results/')[1].split('/')[0]:38s} |combined|={len(combined)}", flush=True)

# --- pre-meta run dirs lacking a replay_state: merge their work/results (v3, delta, ...) ---
premeta_dirs = sorted(d for d in glob.glob(str(BASE / 'results/pentagonbox_*'))
                      if not os.path.exists(os.path.join(d, 'replay_state.pkl'))
                      and os.path.isdir(os.path.join(d, 'work', 'results')))
print(f"[pre-meta] {len(premeta_dirs)} run dirs w/o replay_state -> merge work/results:", flush=True)
for d in premeta_dirs:
    n = merge_workresults(combined, os.path.join(d, 'work', 'results'))
    print(f"  +{n} from {os.path.basename(d):38s} |combined|={len(combined)}", flush=True)

# --- meta cascade caches ---
print("[meta] _resume_cache + _burst_cache + 27 newer reduction.pkls ...", flush=True)
merge_pkl(combined, BASE / 'results/meta_reduce/_resume_cache/replay_state.pkl')
merge_pkl(combined, BASE / 'results/meta_reduce/_burst_cache/replay_state.pkl')
burst = BASE / 'results/meta_reduce/_burst_cache/replay_state.pkl'
newer = subprocess.run(['find', str(BASE / 'results/meta_reduce'), '-name', 'reduction.pkl',
                        '-newer', str(burst)], capture_output=True, text=True).stdout.split()
for p in sorted(newer):
    merge_pkl(combined, p)
print(f"  |combined|={len(combined)}", flush=True)

# --- tgt0028 live ---
n = merge_workresults(combined, str(BASE / 'results/meta_reduce/tgt0028_w7_4/work/results'))
print(f"[tgt0028] +{n}  |combined|={len(combined)}", flush=True)
print(f"\nCOMPLETE CACHE (meta + pre-meta): {len(combined)} entries ({time.time()-t0:.0f}s)\n", flush=True)

# --- replay ---
clean = load_targets(BASE / 'from_federica/list_TA_ispclean_by_weight')
print(f"replaying {len(clean)} ISP-clean list_TA integrals ...\n", flush=True)
t1 = time.time()
fully = 0
residual = []
for idx, t in enumerate(clean):
    expr = apply_substitutions({t: 1}, combined, PRIME)
    nm = [i for i in expr if not is_master(i)]
    if nm:
        residual.append((t, nm))
    else:
        fully += 1
    if (idx + 1) % 200 == 0:
        print(f"  ... {idx+1}/{len(clean)} ({fully} fully, {len(residual)} residual)", flush=True)

print(f"\n========== COMPLETE REPLAY (ISP-clean, {len(clean)}) ==========", flush=True)
print(f"  fully reduced to masters : {fully}/{len(clean)}", flush=True)
print(f"  residual (non-masters)   : {len(residual)}/{len(clean)}  ({time.time()-t1:.0f}s)", flush=True)
for t, nm in residual:
    print(f"\n  RESIDUAL  TA[{','.join(map(str, t))}]  -> {len(nm)} non-master term(s):", flush=True)
    for x in nm[:15]:
        print(f"      {','.join(map(str, x))}", flush=True)
