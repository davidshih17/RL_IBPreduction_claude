#!/usr/bin/env python3
"""onestep_worker_v6.py
=================================================================
Hierarchical-reduction worker that uses `beam_search_v6.beam_search_v5`
internally — strip-passenger + LAZY_RS + iraws-keep-first + tabu +
any() termination + macro-dedup at step start.

CLI mirrors `onestep_worker.py` so `hierarchical_reduction.py` can swap
worker scripts without rewriting the orchestrator. Unsupported flags
(`--n_workers`, `--random-target`, `--dedup-beam-by-content`,
`--beam-sort mixed_dedup`, etc.) are accepted but silently ignored —
v6 has its own dedup at step start and beam_width=40 is single-process.

OUTPUT FORMAT — orchestrator-compatible result.pkl with keys:
    success           bool
    original_integral tuple
    final_expr        dict integral → coeff (start_int in terms of
                      passengers + masters; from v6 path replay)
    path              list of (target, ibp_op, delta) tuples
    restart_offsets   [] (v6 has no restart loop)
    steps             int
    time              float (wall seconds)
    prime             int
    peak_memory_kb    int (ru_maxrss)
"""
import argparse
import os
import pickle
import resource
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, IBPEnvironment, weight,
)
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask, max_weight
from beam_search_v6 import beam_search_v5, replay_full_expr


def main():
    p = argparse.ArgumentParser(
        description='v6-backed onestep worker for hierarchical_reduction.py'
    )
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-checkpoint', required=True)
    p.add_argument('--beam_width', type=int, default=40)
    p.add_argument('--max_steps', type=int, default=10**6)
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--device', default='cpu')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--paper-masters-only',
                   action=argparse.BooleanOptionalAction, default=True)
    # The following are accepted for CLI parity with onestep_worker.py but
    # do not affect v6 (v6 handles dedup/parallelism differently).
    p.add_argument('--n_workers', type=int, default=1)
    p.add_argument('--random-target', action='store_true')
    p.add_argument('--dedup-beam-by-content',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--beam-sort', default='weight',
                   choices=['weight', 'mixed', 'totalweight',
                            'nterms', 'score', 'mixed_dedup'])
    # v6-specific knobs (sensible defaults so orchestrator can omit them).
    p.add_argument('--no-tabu', dest='tabu', action='store_false', default=True)
    p.add_argument('--no-lazy-rs', dest='lazy_rs',
                   action='store_false', default=True)
    p.add_argument('--iraws-keep-first', type=int, default=50)
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=False)
    p.add_argument('--n-threads', type=int, default=1)
    # Accept (and ignore) the checkpoint/resume flags onestep_worker.py uses.
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=50)
    p.add_argument('--checkpoint-time-seconds', type=int, default=300)
    p.add_argument('--resume-from', default=None)
    args = p.parse_args()

    torch.set_num_threads(args.n_threads)

    t0 = time.time()

    # ── topology + prime ──────────────────────────────────────────────
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()

    # ── model load ────────────────────────────────────────────────────
    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ck = torch.load(args.model_checkpoint, map_location=args.device,
                    weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.to(args.device)
    model.eval()

    # ── parse integral + setup ────────────────────────────────────────
    integral_str = args.integral.strip("'\"")
    start_int = tuple(int(x) for x in integral_str.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))
    start_expr = {start_int: 1}

    if args.verbose:
        print(f'[v6-worker] integral I{list(start_int)}', flush=True)
        print(f'[v6-worker] weight={start_w12} target_sector={target_sector}',
              flush=True)
        print(f'[v6-worker] beam_width={args.beam_width} '
              f'max_steps={args.max_steps} tabu={args.tabu} '
              f'iraws_keep_first={args.iraws_keep_first} '
              f'lazy_rs={args.lazy_rs}', flush=True)

    # ── run v6 ────────────────────────────────────────────────────────
    # Checkpoint plumbing: the orchestrator's straggler-resubmit story
    # gives us a checkpoint-path; v6 saves the rolling ckpt there.
    ckpt_path = None
    if args.checkpoint_path and not args.no_checkpoint:
        ckpt_path = args.checkpoint_path

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        beam_sort='weight' if args.beam_sort == 'mixed_dedup' else args.beam_sort,
        ckpt_path=ckpt_path,
        ckpt_every=args.checkpoint_interval,
        verbose=args.verbose,
        resume_from=args.resume_from,
        tabu=args.tabu,
        use_incremental_aux=True,
        use_exprkeyed=args.use_exprkeyed,
        ckpt_every_step=False,
        iraws_window=None,
        iraws_keep_first=args.iraws_keep_first,
        lazy_rs=args.lazy_rs,
    )

    elapsed = time.time() - t0

    success = (best_state.n_non_masters == 0)
    final_mw = max_weight(best_state.expr, target_sector)
    path = list(best_state.path)
    n_steps = len(path)

    # ── reconstruct full expression (with passengers) ─────────────────
    # v6 strip-passenger drops sub-weight terms from the in-memory expr,
    # so we need to replay through the raw IBP templates against the
    # original start_expr to get the orchestrator-expected final_expr.
    full = replay_full_expr(start_expr, path, env)
    if full is not None:
        full_expr = full[0]
    else:
        full_expr = dict(best_state.expr)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        'success': success,
        'original_integral': start_int,
        'final_expr': full_expr,
        'path': path,
        'restart_offsets': [],   # v6 has no restart loop
        'steps': n_steps,
        'time': elapsed,
        'prime': args.prime,
        'peak_memory_kb': peak_rss_kb,
        # v6-specific debug
        'best_n_non_masters': best_state.n_non_masters,
        'best_max_w12': final_mw,
        'start_w12': start_w12,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    if args.verbose:
        status = 'SUCCESS' if success else 'INCOMPLETE'
        print(f'\n[v6-worker] {status} in {elapsed:.2f}s, '
              f'path_len={n_steps}, peak_rss={peak_rss_kb/1024:.1f} MB',
              flush=True)
        print(f'[v6-worker] best_mw={final_mw} nm={best_state.n_non_masters}',
              flush=True)
        print(f'[v6-worker] wrote {args.output}', flush=True)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
