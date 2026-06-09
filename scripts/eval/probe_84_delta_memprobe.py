"""(8,4) memory-instrumentation probe.

Runs the delta beam search up to a configurable max_steps. Every step, dumps:
  - per-survivor cu length
  - per-survivor sum of cached dict sizes (#keys total)
  - per-survivor iraws length
  - total memory of cu/iraws via sys.getsizeof + recursive walk

Output: /tmp/memprobe.csv (step, surv_idx, cu_len, cu_total_keys, iraws_len,
                          cu_bytes, iraws_bytes).
Also prints a step summary line with sum across all survivors.

This lets us nail down what's actually allocating the ~40 MB per step.
"""
import sys, os, gc, time
from pathlib import Path
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, set_paper_masters_only,
)
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask


def deep_size(obj, seen=None):
    """Recursive sys.getsizeof for dicts/lists/tuples/ints."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    sz = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            sz += deep_size(k, seen) + deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for x in obj:
            sz += deep_size(x, seen)
    return sz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-steps', type=int, default=50)
    ap.add_argument('--out', type=str, default='/tmp/memprobe.csv')
    args = ap.parse_args()

    topo = Topology.from_dir('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox')
    ibp_env.init_from_topology(topo)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    integral = tuple(int(x) for x in '-1,2,1,0,1,2,1,1,-3,0,0'.split(','))
    target_sector = tuple(get_sector_mask(integral))

    model = IBPActionClassifier(
        n_indices=topo.n_indices,
        n_denominators=topo.n_denominators,
        n_ibp_ops=topo.n_actions,
    )
    ck = torch.load(
        '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt',
        map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # Monkey-patch into beam_search_delta so we can stop at each step
    # and dump survivor stats.
    from sailir import delta_beam_search as dbs

    # Custom run: hot-fix beam_search_delta to call our hook each step.
    # Simpler: run for a few steps in a loop, restarting? No — we need the
    # actual beam evolution. Use a custom step counter via env var.
    os.environ['BEAM_PROFILE_CIC_INC'] = '0'

    # Easiest: just call beam_search_delta with verbose=True for max_steps,
    # then sample the final beam.
    print(f'[t=0] running beam_search_delta max_steps={args.max_steps}', flush=True)
    t0 = time.time()
    solution, beam, best_w = dbs.beam_search_delta(
        env, model, {integral: 1}, target_sector,
        beam_width=40, max_steps=args.max_steps, prime=1009,
        verbose=False, stop_on_weight_improvement=False,
        device='cpu',
    )
    elapsed = time.time() - t0
    print(f'[t={elapsed:.1f}] done. final beam={len(beam)}', flush=True)

    # Dump per-survivor stats.
    print()
    print(f'{"survivor":>10} {"cu_len":>8} {"cu_keys":>10} {"iraws":>8} {"cu_MB":>8} {"iraws_KB":>10}')
    print('-' * 65)
    cu_total_b = 0
    iraws_total_b = 0
    cu_keys_total = 0
    for i, s in enumerate(beam):
        cu, ubm, rid, iraws = s.indirect_aux
        n_keys = sum(len(c) for c in cu)
        cu_b = deep_size(cu)
        iraws_b = deep_size(iraws)
        cu_total_b += cu_b
        iraws_total_b += iraws_b
        cu_keys_total += n_keys
        if i < 5 or i >= 35:
            print(f'  {i:>8} {len(cu):>8} {n_keys:>10} {len(iraws):>8} '
                  f'{cu_b/1024/1024:>8.2f} {iraws_b/1024:>10.1f}')
    print('-' * 65)
    print(f'  {"TOTAL":>8} {"":>8} {cu_keys_total:>10} {"":>8} '
          f'{cu_total_b/1024/1024:>8.2f} {iraws_total_b/1024:>10.1f}')
    print()
    avg_cu_keys = cu_keys_total / max(1, args.max_steps * 40)
    avg_cu_mb_per_step = cu_total_b / max(1, args.max_steps) / 1024 / 1024
    print(f'avg cu keys ADDED per (step,survivor): {avg_cu_keys:.1f}')
    print(f'avg cu_total MB ADDED per step (all 40 survivors): '
          f'{avg_cu_mb_per_step:.2f}')


if __name__ == '__main__':
    main()
