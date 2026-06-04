#!/usr/bin/env python
"""Diff two runs' thick per-step checkpoints, find first step where any
field of any beam survivor differs. Reports EVERY field that differs.

Compares for each survivor at each step:
  - expr
  - subs
  - resolved_subs
  - aux_flat.cu (cu_keys + cu_coeffs + cu_lengths)
  - aux_flat.iraws_meta (sub_int, op, shift)
  - aux_flat.ubm
  - max_w, n_non_masters, score, path_len, path

Beam comparison is SET-LEVEL: each survivor in A is matched against a
survivor in B with identical (path) — they should pair exactly if the
runs are deterministic up to this step. Mismatched pairs = divergence.

Usage:
  diff_thick_ckpts.py <dir_a> <dir_b> [--start-step N] [--max-step N]
"""
import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_aux_flat(flat):
    """Convert FlatAux to plain dicts for comparison."""
    if flat is None:
        return None
    return {
        'n_cu': int(flat.n_cu),
        'n_iraws': int(flat.n_iraws),
        'cu_offsets': flat.cu_offsets.tolist(),
        'cu_lengths': flat.cu_lengths.tolist(),
        'cu_keys': flat.cu_keys.tolist(),
        'cu_coeffs': flat.cu_coeffs.tolist(),
        'ubm': flat.ubm.tolist(),
        'iraws_meta': flat.iraws_meta.tolist(),
    }


def compare_survivor(sa, sb):
    """Return list of differing field names. Empty = identical."""
    diffs = []
    for k in ('max_w', 'n_non_masters', 'score', 'path_len', 'path',
              'expr', 'subs', 'resolved_subs'):
        va, vb = sa.get(k), sb.get(k)
        if va != vb:
            diffs.append(k)
    fa = normalize_aux_flat(sa.get('aux_flat'))
    fb = normalize_aux_flat(sb.get('aux_flat'))
    if fa is None and fb is None:
        pass
    elif fa is None or fb is None:
        diffs.append('aux_flat:presence')
    else:
        for k in ('n_cu', 'n_iraws', 'cu_offsets', 'cu_lengths',
                  'cu_keys', 'cu_coeffs', 'ubm', 'iraws_meta'):
            if fa[k] != fb[k]:
                diffs.append(f'aux_flat.{k}')
    return diffs


def survivor_key(s):
    """Identity for matching across runs: path is the canonical lineage."""
    return tuple(tuple(a) for a in s['path'])


def step_files(d):
    """Return dict {step_int: path} from result.pkl.ckpt.r1.stepNNNN files."""
    out = {}
    for p in sorted(Path(d).glob('result.pkl.ckpt.r1.step*')):
        suffix = p.name.split('.step', 1)[1]
        try:
            out[int(suffix)] = p
        except ValueError:
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir_a')
    p.add_argument('dir_b')
    p.add_argument('--start-step', type=int, default=1)
    p.add_argument('--max-step', type=int, default=10000)
    args = p.parse_args()

    files_a = step_files(args.dir_a)
    files_b = step_files(args.dir_b)
    common_steps = sorted(set(files_a) & set(files_b))
    common_steps = [s for s in common_steps
                    if args.start_step <= s <= args.max_step]
    print(f'A has {len(files_a)} step ckpts, B has {len(files_b)}, '
          f'common range {min(common_steps) if common_steps else "-"} '
          f'to {max(common_steps) if common_steps else "-"}', flush=True)

    # Track FIRST divergence step per field across all common steps.
    # Keep scanning so we can pinpoint when expr/resolved_subs first
    # diverge even after iraws-only diffs have started.
    first_div = {}
    field_groups = {
        'max_w': 'sort',
        'n_non_masters': 'sort',
        'score': 'sort',
        'path_len': 'sort',
        'path': 'sort',
        'expr': 'search',
        'subs': 'search',
        'resolved_subs': 'search',
        'aux_flat.n_cu': 'aux',
        'aux_flat.n_iraws': 'aux',
        'aux_flat.cu_offsets': 'aux',
        'aux_flat.cu_lengths': 'aux',
        'aux_flat.cu_keys': 'aux',
        'aux_flat.cu_coeffs': 'aux',
        'aux_flat.ubm': 'aux',
        'aux_flat.iraws_meta': 'aux',
        'aux_flat:presence': 'aux',
        '_beam_path_set': 'beam',
    }

    for step in common_steps:
        ca = load(files_a[step])
        cb = load(files_b[step])

        beam_a = ca['beam']
        beam_b = cb['beam']
        if len(beam_a) != len(beam_b):
            print(f'STEP {step}: beam size differs '
                  f'|A|={len(beam_a)} |B|={len(beam_b)}', flush=True)
            return 1

        by_path_a = {survivor_key(s): s for s in beam_a}
        by_path_b = {survivor_key(s): s for s in beam_b}
        paths_a = set(by_path_a)
        paths_b = set(by_path_b)
        a_only = paths_a - paths_b
        b_only = paths_b - paths_a
        common_paths = paths_a & paths_b

        if a_only or b_only:
            if '_beam_path_set' not in first_div:
                first_div['_beam_path_set'] = step
                print(f'\n*** FIRST BEAM-PATH-SET DIVERGENCE AT STEP {step} ***',
                      flush=True)
                print(f'  A_only={len(a_only)} B_only={len(b_only)} '
                      f'common={len(common_paths)}', flush=True)
                for i, p in enumerate(sorted(a_only)[:2]):
                    print(f'  A-only #{i} last3 = {p[-3:]}', flush=True)
                for i, p in enumerate(sorted(b_only)[:2]):
                    print(f'  B-only #{i} last3 = {p[-3:]}', flush=True)

        # For paths in both, compare every field. Aggregate which fields
        # diverged across all common-path survivors at this step.
        step_field_set = set()
        sample_diff_path = None
        for p in common_paths:
            sa = by_path_a[p]
            sb = by_path_b[p]
            d = compare_survivor(sa, sb)
            if d:
                step_field_set.update(d)
                if sample_diff_path is None:
                    sample_diff_path = p

        for f in step_field_set:
            if f not in first_div:
                first_div[f] = step
                grp = field_groups.get(f, '?')
                # Find the specific survivor where THIS field diverges.
                specific = None
                for p in common_paths:
                    d = compare_survivor(by_path_a[p], by_path_b[p])
                    if f in d:
                        specific = p
                        break
                print(f'\n*** First {grp.upper()} divergence: field "{f}" '
                      f'at STEP {step} ***', flush=True)
                if specific is not None:
                    sa = by_path_a[specific]
                    sb = by_path_b[specific]
                    print(f'  diverging survivor: max_w_A={sa["max_w"]} '
                          f'nm_A={sa["n_non_masters"]} score_A={sa["score"]!r} '
                          f'| max_w_B={sb["max_w"]} nm_B={sb["n_non_masters"]} '
                          f'score_B={sb["score"]!r}', flush=True)
                    print(f'  path[-3:]={specific[-3:]}', flush=True)

        # Always emit per-step summary: # paths in common, # field diffs
        any_diff_this_step = bool(step_field_set) or bool(a_only) or bool(b_only)
        n_div = len(first_div)
        flag = '\u26a0' if any_diff_this_step else '.'
        print(f'  step {step:4d} [{flag}]: common={len(common_paths)} '
              f'A_only={len(a_only)} B_only={len(b_only)} '
              f'field_diffs_this_step={len(step_field_set)} '
              f'total_first_divs={n_div}', flush=True)

    print(f'\nScanned {len(common_steps)} common steps.')
    print(f'\n=== First-divergence summary by field ===')
    for f, st in sorted(first_div.items(), key=lambda kv: (kv[1], kv[0])):
        grp = field_groups.get(f, '?')
        print(f'  step {st:4d} [{grp:6s}]  {f}')

    if not first_div:
        print('  (all common steps bit-identical across all fields)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
