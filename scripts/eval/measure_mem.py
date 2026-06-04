#!/usr/bin/env python
"""Measure actual peak RSS while doing one full model forward pass on the
step-166 batch at varying max_actions. Reports per-stage RSS.
"""
import argparse
import os
import pickle
import resource
import sys
import time

import torch

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, enumerate_valid_actions_with_indirect_cache,
    cached_union_bitmask, weight,
)
from beam_search_full import prepare_batched_input_v5
from beam_search_utils import get_non_masters
from sailir.classifier import IBPActionClassifier


def rss_mb():
    # /proc/self/status VmRSS for current process memory; works on Linux.
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0  # KB -> MB
    return float('nan')


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stage(label, before=None):
    cur = rss_mb()
    pk = peak_mb()
    if before is None:
        print(f'[{label:35s}] RSS={cur:>8.1f} MB  peak={pk:>8.1f} MB')
    else:
        delta = cur - before
        print(f'[{label:35s}] RSS={cur:>8.1f} MB  peak={pk:>8.1f} MB  '
              f'Δ={delta:+7.1f} MB')
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('topology')
    p.add_argument('--model', required=True)
    p.add_argument('--max-actions', type=int, default=8000)
    args = p.parse_args()

    torch.set_num_threads(1)

    print(f'== Measuring peak RSS at max_actions={args.max_actions} ==')
    r0 = stage('startup')

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()
    r1 = stage('after env init', r0)

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    r2 = stage('after model load', r1)

    with open(args.ckpt, 'rb') as f:
        c = pickle.load(f)
    target_sector = tuple(c['target_sector'])
    print(f'  loaded checkpoint at step {c["step"]}, beam_size={len(c["beam"])}')
    r3 = stage('after ckpt load', r2)

    # Build the full task list as production would (state, target, valid)
    n_idx = ibp_env.N_INDICES
    tasks = []  # (expr, subs, valid_actions, target_sector, target)
    for s in c['beam']:
        nm = get_non_masters(s['expr'], target_sector)
        if not nm:
            continue
        mw = tuple(s['max_w'])
        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
        if not tied:
            continue
        # Build iraws + cached
        iraws = []
        for row in s['aux_flat'].iraws_meta:
            sub_int = tuple(int(x) for x in row[:n_idx])
            op = int(row[n_idx])
            shift = tuple(int(x) for x in row[n_idx + 1:])
            seed = tuple(sub_int[i] - shift[i] for i in range(n_idx))
            raw = env.get_raw_equation_cached(op, seed)
            cached = apply_resolved_subs(raw, s['resolved_subs'])
            ub = cached_union_bitmask(cached)
            iraws.append((sub_int, op, shift, raw, cached, ub))
        for target in tied:
            va = enumerate_valid_actions_with_indirect_cache(
                target, iraws, s['subs'], s['resolved_subs'],
                env.ibp_t, env.li_t, env.shifts, 'subsector', env._raw_eq_cache,
            )
            tasks.append((s['expr'], s['subs'], va, target_sector, target))

    print(f'  built {len(tasks)} tasks; valid len min={min(len(t[2]) for t in tasks)} '
          f'max={max(len(t[2]) for t in tasks)} mean={sum(len(t[2]) for t in tasks)/len(tasks):.0f}')
    r4 = stage('after building tasks (iraws,cached,valid)', r3)

    # Run the model
    print(f'  running model with current beam_search_full.max_actions cap '
          f'(currently {args.max_actions})...')
    t0 = time.time()
    batch, _ = prepare_batched_input_v5(tasks, 'cpu')
    r5 = stage('after prepare_batched_input_v5', r4)
    print(f'  action tensor shape: {tuple(batch["action_ibp_ops"].shape)}')

    with torch.no_grad():
        logits, probs = model(
            batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
            batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
            batch['sub_repl_mask'], batch['sub_mask'],
            batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
            batch['sector_mask'], batch['target_integral'],
        )
    t1 = time.time()
    r6 = stage('after model forward', r5)
    print(f'  model time: {(t1-t0):.2f}s')
    print(f'  logits shape: {tuple(logits.shape)}')
    print(f'\n== FINAL peak RSS: {peak_mb():.1f} MB ==')


if __name__ == '__main__':
    sys.exit(main())
