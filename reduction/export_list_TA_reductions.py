#!/usr/bin/env python3
"""Export the integral-by-integral reduction of all 996 ISP-clean list_TA
integrals to the master basis.

Basis: pentagonbox topology, --no-paper-masters-only, over GF(1009) at the fixed
kinematic point. Each result is a linear combination of master integrals with
integer coefficients mod 1009.

Writes (to results/meta_reduce/):
  list_TA_reductions.txt  — human-readable, one block per integral
  list_TA_reductions.pkl  — {integral_tuple: {master_tuple: coeff}}  (machine)
  list_TA_masters.txt     — the distinct master set that appears across all 996
"""
import pickle, re, sys, glob, os, subprocess, time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'reduction'))
from sailir.topology import Topology
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only, is_master, weight
from beam_search_utils import get_sector_mask

PRIME = 1009
init_from_topology(Topology.from_dir(str(BASE / 'topology_input/pentagonbox')))
set_prime(PRIME); set_paper_masters_only(False)
OUT = BASE / 'results/meta_reduce'


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


def fmt(t):
    return '[' + ','.join(map(str, t)) + ']'


def mkey(m):  # sort masters: higher sector/level/weight first, then lexicographic
    L = sum(get_sector_mask(m))
    r, s = weight(m)[:2]
    return (-L, -r, -s, m)


# ---- build the final combined cache (same sources as replay_list_TA_final.py) ----
t0 = time.time()
combined = {}
R = BASE / 'results'
merge_pkl(combined, R / 'pentagonbox_8_5_v7_fresh/replay_state.pkl')
merge_wr(combined, str(R / 'pentagonbox_8_5_v7_fresh_round2/work/results'))
merge_pkl(combined, R / 'pentagonbox_7_5_cache_from_85/replay_state.pkl')
merge_pkl(combined, R / 'pentagonbox_7_6_cache_from_85_75/replay_state.pkl')
burst = R / 'meta_reduce/_burst_cache/replay_state.pkl'
merge_pkl(combined, burst)
for p in sorted(subprocess.run(['find', str(R / 'meta_reduce'), '-name', 'reduction.pkl', '-newer', str(burst)],
                               capture_output=True, text=True).stdout.split()):
    merge_pkl(combined, p)
for n in ['tgt0028_w7_4', 'extra0001_w6_4', 'extra0002_w6_3', 'extra0003_w6_3']:
    merge_pkl(combined, R / f'meta_reduce/{n}/reduction.pkl')
print(f"combined cache: {len(combined)} entries ({time.time()-t0:.0f}s)", flush=True)

# ---- reduce every ISP-clean integral, in the file's (decreasing-weight) order ----
targets = []
for ln in open(BASE / 'from_federica/list_TA_ispclean_by_weight'):
    m = re.match(r'TA\[([0-9,\-]+)\]', ln.strip())
    if m:
        targets.append(tuple(int(x) for x in m.group(1).split(',')))

reductions = {}        # integral -> {master: coeff}
masters_seen = set()
bad = []
for t in targets:
    expr = apply_substitutions({t: 1}, combined, PRIME)
    if any(not is_master(i) for i in expr):
        bad.append(t)
    reductions[t] = expr
    masters_seen.update(expr.keys())

print(f"reduced {len(targets)} integrals; {len(bad)} with residual non-masters; "
      f"{len(masters_seen)} distinct masters across all", flush=True)

# ---- write machine-readable ----
with open(OUT / 'list_TA_reductions.pkl', 'wb') as f:
    pickle.dump({'basis': 'pentagonbox / no-paper-masters-only', 'prime': PRIME,
                 'reductions': reductions, 'masters': sorted(masters_seen, key=mkey)}, f)

# ---- write human-readable ----
with open(OUT / 'list_TA_reductions.txt', 'w') as f:
    f.write(f"# list_TA reduction to master integrals\n")
    f.write(f"# basis = pentagonbox, --no-paper-masters-only ; field = GF({PRIME})\n")
    f.write(f"# {len(targets)} ISP-clean integrals -> {len(masters_seen)} distinct masters\n")
    f.write(f"# coefficients are integers mod {PRIME}; masters written as TA[...]\n\n")
    for t in targets:
        r, s = weight(t)[:2]
        expr = reductions[t]
        ms = sorted(expr.items(), key=lambda kv: mkey(kv[0]))
        f.write(f"TA{fmt(t)}  (r={r},s={s})  =  {len(ms)} master(s)\n")
        for mst, c in ms:
            f.write(f"    {c:>4} * TA{fmt(mst)}\n")
        f.write("\n")

with open(OUT / 'list_TA_masters.txt', 'w') as f:
    f.write(f"# {len(masters_seen)} distinct master integrals (basis = pentagonbox / no-paper)\n")
    for mst in sorted(masters_seen, key=mkey):
        L = sum(get_sector_mask(mst)); r, s = weight(mst)[:2]
        f.write(f"TA{fmt(mst)}  (L={L},r={r},s={s})\n")

print(f"\nwrote:\n  {OUT/'list_TA_reductions.txt'}\n  {OUT/'list_TA_reductions.pkl'}\n  {OUT/'list_TA_masters.txt'}", flush=True)

# ---- sample to stdout ----
print("\n===== SAMPLE (first 2 integrals) =====", flush=True)
for t in targets[:2]:
    expr = reductions[t]
    ms = sorted(expr.items(), key=lambda kv: mkey(kv[0]))
    print(f"TA{fmt(t)}  =  {len(ms)} masters:", flush=True)
    for mst, c in ms[:8]:
        print(f"    {c:>4} * TA{fmt(mst)}", flush=True)
    if len(ms) > 8:
        print(f"    ... (+{len(ms)-8} more)", flush=True)
