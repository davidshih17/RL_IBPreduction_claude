#!/usr/bin/env python3
"""Compare Option F probe_74 against split-expr-with-sub_accum baseline.

Loads both result.pkl files and verifies the FULL final_expr (target-sector
AND sub-sector content) is bit-identical. Also reports path length and step
count for completeness.

Baseline: probe_74_no_dedup_with_ckpt (cluster 1468428), split-expr with
sub_accum tracked through every beam_search step.
Option F: probe_74_optionF (cluster 1469793), target-only substitution
during beam_search + path replay reconstruction at worker end.

Both must produce identical full final_expr if Option F is correct.
"""
import argparse
import pickle
import sys
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results')


def load(p):
    if not p.exists():
        print(f"ERROR: {p} does not exist", file=sys.stderr)
        sys.exit(1)
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str,
                        default=str(BASE / 'probe_74_no_dedup_with_ckpt/result.pkl'),
                        help='Baseline result.pkl (default: probe_74_no_dedup_with_ckpt)')
    parser.add_argument('--candidate', type=str,
                        default=str(BASE / 'probe_74_optionF/result.pkl'),
                        help='Candidate result.pkl to compare against baseline')
    args = parser.parse_args()
    BASELINE = Path(args.baseline)
    OPTIONF = Path(args.candidate)

    a = load(BASELINE)
    b = load(OPTIONF)

    print(f"Baseline pkl: {BASELINE}")
    print(f"  success={a['success']} steps={a['steps']} path_len={len(a['path'])}")
    print(f"  final_expr size: {len(a['final_expr'])}")
    print(f"  time={a['time']:.1f}s")
    print()
    print(f"Option F pkl: {OPTIONF}")
    print(f"  success={b['success']} steps={b['steps']} path_len={len(b['path'])}")
    print(f"  final_expr size: {len(b['final_expr'])}")
    print(f"  time={b['time']:.1f}s")
    print()

    # 1) Path equality (paths should be identical step-for-step under same seed).
    same_path = (a['path'] == b['path'])
    print(f"Path identical? {same_path}")
    if not same_path:
        if len(a['path']) != len(b['path']):
            print(f"  Path lengths differ: baseline={len(a['path'])}, optionF={len(b['path'])}")
        # Find first divergence
        for i, (sa, sb) in enumerate(zip(a['path'], b['path'])):
            if sa != sb:
                print(f"  First divergence at step {i}:")
                print(f"    baseline: {sa}")
                print(f"    optionF:  {sb}")
                break

    # 2) Full final_expr equality — the load-bearing check for Option F.
    ea = a['final_expr']
    eb = b['final_expr']
    same_expr = (ea == eb)
    print(f"\nFull final_expr identical? {same_expr}")
    if not same_expr:
        ka = set(ea.keys()); kb = set(eb.keys())
        only_a = ka - kb
        only_b = kb - ka
        common = ka & kb
        diff_vals = {k for k in common if ea[k] != eb[k]}
        print(f"  baseline-only keys: {len(only_a)}")
        print(f"  optionF-only keys:  {len(only_b)}")
        print(f"  common keys with different coeffs: {len(diff_vals)}")
        for k in list(only_a)[:5]:
            print(f"    baseline-only: I{list(k)} -> {ea[k]}")
        for k in list(only_b)[:5]:
            print(f"    optionF-only:  I{list(k)} -> {eb[k]}")
        for k in list(diff_vals)[:5]:
            print(f"    diff: I{list(k)} baseline={ea[k]} optionF={eb[k]}")
        sys.exit(1)

    print("\nSUCCESS: Option F produces bit-identical full final_expr (target + sub-sector).")


if __name__ == '__main__':
    main()
