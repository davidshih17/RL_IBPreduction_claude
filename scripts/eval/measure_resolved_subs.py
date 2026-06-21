#!/usr/bin/env python3
"""Decisive (mostly zero-cost) proof of the memory culprit. For each keep_step
beam snapshot in a probe dir, measure the serialized byte-size of each State
field summed over ALL beam states: resolved_subs vs expr vs aux_flat. Show
resolved_subs dominates and scales ~linearly with rs_vsz, then extrapolate to the
real stuck job's end point (rs_vsz=77155 -> MemoryUsage 21.5 GB) to confirm.

Serialized (pickle) bytes are a faithful proxy: resolved_subs values are PackedEq
(int32/int16 numpy arrays) whose pickle size ~= array nbytes. Usage:
    python measure_resolved_subs.py <probe_subdir>     (default: probe_stuck_74_ckpt)
"""
import glob
import pickle
import sys

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + '/scripts/eval')
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import init_from_topology, set_prime, set_paper_masters_only

_arg = sys.argv[1] if len(sys.argv) > 1 else 'probe_stuck_74_ckpt'
D = _arg if _arg.startswith('/') else f'{BASE}/results/{_arg}'

init_from_topology(Topology.from_dir(f'{BASE}/topology_input/pentagonbox'))
set_prime(1009)
set_paper_masters_only(False)


def pbytes(obj):
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


def rs_count(rs):
    """rs_vsz-style entry count = sum(len(v) for v in resolved_subs.values())."""
    tot = 0
    for v in rs.values():
        try:
            tot += len(v)
        except TypeError:
            tot += len(getattr(v, 'ids', ()))   # PackedEq fallback
    return tot


def load(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
        states = [pickle.load(f) for _ in range(meta['n_states'])]
    return meta, states


files = sorted(glob.glob(f'{D}/ckpt.pkl.keep_step*'),
               key=lambda p: int(p.rsplit('keep_step', 1)[-1]))
if not files:
    print(f'no keep_step snapshots in {D}', flush=True)
    sys.exit(1)

print(f'{D}  ({len(files)} snapshots)\n', flush=True)
print(f"{'step':>5} {'rs_vsz(best)':>12} {'RS_bytes_all':>14} "
      f"{'expr_bytes_all':>14} {'aux_bytes_all':>14} {'RS%':>5} {'B/entry':>8}",
      flush=True)
rows = []
for path in files:
    meta, states = load(path)
    rs_b = sum(pbytes(s['resolved_subs']) for s in states if s.get('resolved_subs'))
    ex_b = sum(pbytes(s['expr']) for s in states)
    ax_b = sum(pbytes(s.get('aux_flat')) for s in states if s.get('aux_flat'))
    best = states[0]
    rsv = rs_count(best['resolved_subs']) if best.get('resolved_subs') else 0
    rsv_all = sum(rs_count(s['resolved_subs']) for s in states
                  if s.get('resolved_subs'))
    tot = rs_b + ex_b + ax_b
    pct = 100.0 * rs_b / tot if tot else 0
    bpe = rs_b / rsv_all if rsv_all else 0
    rows.append((meta['step'], rsv, rsv_all, rs_b, ex_b, ax_b))
    print(f"{meta['step']:5d} {rsv:12d} {rs_b:14d} {ex_b:14d} {ax_b:14d} "
          f"{pct:4.0f}% {bpe:8.1f}", flush=True)

# Extrapolate RS bytes (summed over all states) to the real end point.
# Fit RS_bytes_all = k * rsv_all (proportional through ~0).
import statistics
ks = [r[3] / r[2] for r in rows if r[2] > 0]      # rs_b / rsv_all
k = statistics.median(ks)
END_RSV_BEST = 77155                                # stuck job 1663961 @ step 1830
# rsv_all ~ beam_width * rsv_best (best is representative); use observed ratio.
ratio = statistics.median([r[2] / r[1] for r in rows if r[1] > 0])
end_rsv_all = END_RSV_BEST * ratio
pred_gb = k * end_rsv_all / 1024**3
print(f"\nmedian bytes/entry (all-states RS) = {k:.1f}", flush=True)
print(f"observed rsv_all / rsv_best ratio  = {ratio:.1f} (~beam_width)", flush=True)
print(f"EXTRAPOLATION to stuck-job end (rs_vsz_best=77155):", flush=True)
print(f"  predicted resolved_subs RSS ~= {pred_gb:.1f} GB", flush=True)
print(f"  measured job MemoryUsage      = 21.5 GB", flush=True)
print(f"  => resolved_subs explains {100*pred_gb/21.5:.0f}% of the footprint",
      flush=True)
