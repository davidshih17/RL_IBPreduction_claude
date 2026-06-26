#!/usr/bin/env python3
"""Replay every integral in the ISP-clean list_TA subset (997) through the full
cascade cache to CONFIRM each reduces to masters -- i.e. how many of the
original list are fully reduced and which remain (expected: only tgt0028, still
running).

Combined cache (complete, but cheap): the two big standalone caches
_resume_cache (Jun 25) + _burst_cache (Jun 24 21:02), PLUS every reduction.pkl
newer than _burst_cache (the post-21:02 coverage gap -- each embeds a full
~137k-entry 'cache'), PLUS tgt0028's live work/results. Sources are unioned;
a reduction terminates at the master basis regardless of which intermediate
path is kept, so union order is irrelevant for the all-masters test.

is_master uses the pentagonbox topology + paper_masters_only=False to match the
cascade's --no-paper-masters-only basis exactly.
"""
import pickle, re, sys, glob, os, subprocess, time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'reduction'))

from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, is_master

PRIME = 1009
topology = Topology.from_dir(str(BASE / 'topology_input/pentagonbox'))
init_from_topology(topology)
set_prime(PRIME)
set_paper_masters_only(False)   # match the cascade (--no-paper-masters-only)
print(f"topology=pentagonbox prime={PRIME} paper_masters_only=False", flush=True)


def cache_of(obj):
    return obj['cache'] if isinstance(obj, dict) and 'cache' in obj else obj


def merge_pkl(combined, path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  SKIP {os.path.basename(path)}: {e}", flush=True)
        return
    c = cache_of(obj)
    if isinstance(c, dict):
        combined.update(c)
    del obj, c


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
            print("  WARN apply_substitutions hit 10000 iters", flush=True)
            break
    return expr


# ---------- build the combined cache ----------
t0 = time.time()
combined = {}
burst = BASE / 'results/meta_reduce/_burst_cache/replay_state.pkl'
resume = BASE / 'results/meta_reduce/_resume_cache/replay_state.pkl'

print("[merge] _resume_cache ...", flush=True)
merge_pkl(combined, resume); print(f"  |combined|={len(combined)}", flush=True)
print("[merge] _burst_cache ...", flush=True)
merge_pkl(combined, burst); print(f"  |combined|={len(combined)}", flush=True)

newer = subprocess.run(
    ['find', str(BASE / 'results/meta_reduce'), '-name', 'reduction.pkl', '-newer', str(burst)],
    capture_output=True, text=True).stdout.split()
print(f"[merge] {len(newer)} reduction.pkls newer than _burst_cache ...", flush=True)
for i, p in enumerate(sorted(newer)):
    merge_pkl(combined, p)
    if (i + 1) % 10 == 0 or i == len(newer) - 1:
        print(f"  ... {i+1}/{len(newer)}  |combined|={len(combined)}", flush=True)

print("[merge] tgt0028 live work/results ...", flush=True)
nadd = merge_workresults(combined, str(BASE / 'results/meta_reduce/tgt0028_w7_4/work/results'))
print(f"  +{nadd} from tgt0028 work/results  |combined|={len(combined)}", flush=True)
print(f"\nCOMBINED CACHE: {len(combined)} entries  (built in {time.time()-t0:.1f}s)\n", flush=True)

# ---------- replay ----------
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
        print(f"  ... {idx+1}/{len(clean)} replayed ({fully} fully, {len(residual)} residual)", flush=True)

print(f"\n========== REPLAY RESULT (ISP-clean list_TA, {len(clean)}) ==========", flush=True)
print(f"  fully reduced to masters : {fully}/{len(clean)}", flush=True)
print(f"  residual (non-masters)   : {len(residual)}/{len(clean)}  (replay took {time.time()-t1:.1f}s)", flush=True)
for t, nm in residual:
    print(f"\n  RESIDUAL  TA[{','.join(map(str, t))}]  -> {len(nm)} non-master term(s):", flush=True)
    for x in nm[:15]:
        print(f"      {','.join(map(str, x))}", flush=True)
    if len(nm) > 15:
        print(f"      ... (+{len(nm)-15} more)", flush=True)
